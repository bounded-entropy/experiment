"""Delta W = U C(z) V transpose, with learned, shared column directions.

Recorded sequence noise routes a task posterior; the decoder sees z only.
All shared parameters freeze during new-task adaptation. Existing replay
wrappers, Gaussian KL, optimizer groups, and checkpoint framing do the rest.
"""

from dataclasses import dataclass, field
import json

import torch
from torch.nn.functional import normalize
from safetensors.torch import save as st_save

from rlstack.policy.adapters.lora_torch import LoraState, _site_seed
from rlstack.policy.adapters.plora_torch import analytic_kl
from rlstack.policy.adapters.replay import ReplayRows, SiteWrapper, join_site, leave_site, leaf_module
from rlstack.policy.adapters.spectral_latent_torch import unpack
from rlstack.policy.adapters.spectral_tasks import EPS_RECORD, TASK_RECORD


@dataclass
class LearnedTaskState:
    rank: int
    latent: int
    hidden: int
    scope: str
    train_decoder: bool
    scales: str
    beta: float
    mu: torch.nn.Parameter
    log_std: torch.nn.Parameter
    trunk: torch.nn.Sequential
    heads: dict[str, torch.nn.Linear]
    u: dict[str, torch.nn.Parameter]
    v: dict[str, torch.nn.Parameter]
    active_tasks: torch.Tensor | None = field(default=None, repr=False)

    def shared_parameters(self):
        return [*self.trunk.parameters(),
                *(p for head in self.heads.values() for p in head.parameters()),
                *self.u.values(), *self.v.values()]

    def parameters(self):
        return [p for group in param_groups(self).values() for p in group]


def build(sites, init) -> LearnedTaskState:
    """Nonzero learned directions and zero cores give exact base behavior."""
    rank, latent, hidden, tasks = (int(init[k]) for k in ("rank", "latent", "hidden", "tasks"))
    if min(rank, latent, hidden, tasks) < 1 or not sites:
        raise ValueError("positive dimensions and at least one site required")
    if any(s.shape is None or rank > min(s.shape) for s in sites):
        raise ValueError("rank must fit every weighted site")
    scales = init.get("scales", "learned")
    if scales not in ("learned", "fixed", "deterministic"):
        raise ValueError(f"unknown posterior scales: {scales}")
    if init.get("prior", "fixed") != "fixed" or init.get("prior_std", 1.0) != 1.0:
        raise ValueError("learned_tasks uses the fixed N(0,I) prior")
    seed = int(init.get("seed", 0))
    heads, u, v = {}, {}, {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        trunk = torch.nn.Sequential(torch.nn.Linear(latent, hidden), torch.nn.SiLU(),
                                    torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        for site in sites:
            d_in, d_out = site.shape
            generator = torch.Generator().manual_seed(_site_seed(seed, site.path))
            u[site.path] = torch.nn.Parameter(normalize(torch.randn(d_out, rank, generator=generator), dim=0))
            v[site.path] = torch.nn.Parameter(normalize(torch.randn(d_in, rank, generator=generator), dim=0))
            head = torch.nn.Linear(hidden, rank * rank)
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
            heads[site.path] = head
    state = LearnedTaskState(rank, latent, hidden, str(init["scope"]),
        bool(init.get("train_decoder", True)), scales, float(init.get("beta", 1e-3)),
        torch.nn.Parameter(torch.zeros(tasks, latent)),
        torch.nn.Parameter(torch.zeros(tasks, latent), requires_grad=scales == "learned"),
        trunk, heads, u, v)
    for p in state.shared_parameters():
        p.requires_grad_(state.train_decoder)
    return state


def param_groups(state):
    groups = {"mean": [state.mu]}
    if state.scales == "learned":
        groups["scale"] = [state.log_std]
    if state.train_decoder:
        groups["mapper"] = state.shared_parameters()
    return groups


def task_latents(state: LearnedTaskState, rows: ReplayRows):
    """The sealed sequence owns one task index and one draw across all sites."""
    if rows.facts is None:
        raise ValueError("learned_tasks requires recorded tasks and Gaussian noise")
    indices, noise = [], []
    for turns in rows.facts:
        tasks = {int(turn[TASK_RECORD]) for turn in turns}
        draws = {tuple(turn[EPS_RECORD]) for turn in turns}
        if len(tasks) != 1 or len(draws) != 1:
            raise ValueError("one task and one draw per sequence required")
        task, eps = tasks.pop(), draws.pop()
        if not 0 <= task < len(state.mu) or len(eps) != state.latent:
            raise ValueError("task or noise does not fit the posterior bank")
        indices.append(task)
        noise.append(eps)
    state.active_tasks = torch.tensor(indices, device=state.mu.device)
    mu = state.mu[state.active_tasks]
    if state.scales == "deterministic":
        return mu
    eps = torch.tensor(noise, dtype=mu.dtype, device=mu.device)
    return mu + state.log_std[state.active_tasks].exp() * eps


def cores(state: LearnedTaskState, path: str, z: torch.Tensor):
    return state.heads[path](state.trunk(z)).reshape(-1, state.rank, state.rank)


class LearnedTaskSite(SiteWrapper):
    def forward(self, x):
        rows = self.plan.rows
        slot = rows.uniform()
        if slot is None:
            raise ValueError("one tenant per forward required")
        state = slot.get(self.path)
        if not isinstance(state, LearnedTaskState):
            return self.inner(x)
        z = task_latents(state, rows)
        if x.dim() != 3 or x.shape[0] != z.shape[0]:
            raise ValueError("task adapters require [sequences,tokens,width]")
        u, v = normalize(state.u[self.path], dim=0), normalize(state.v[self.path], dim=0)
        core = cores(state, self.path, z)
        delta = torch.bmm(x.float() @ v, core.transpose(1, 2)) @ u.T
        return self.inner(x) + delta.to(x.dtype)


def provide(state):
    ids = state.active_tasks
    if ids is None:
        ids = torch.arange(len(state.mu), device=state.mu.device)
    mu, log_std = state.mu[ids], state.log_std[ids]
    penalty = (0.5 * mu.square().sum() if state.scales == "deterministic"
               else analytic_kl(mu, log_std, mu.new_tensor(0.0))) / len(ids)
    return {"latent_kl": penalty, "factual_beta": mu.new_tensor(state.beta),
            "posterior_std": (log_std.exp().mean() if state.scales != "deterministic"
                              else mu.new_tensor(0.0))}


def install(model, state):
    for path in state.heads:
        parent, leaf = leaf_module(model, path)
        device = next(getattr(parent, leaf).parameters()).device
        for p in (state.mu, state.log_std, *state.shared_parameters()):
            p.data = p.data.to(device)
        join_site(model, path, LearnedTaskSite, state)


def uninstall(model, state):
    for path in state.heads:
        leave_site(model, path, LearnedTaskSite, state)


def shared_tensors(state):
    tensors = {f"trunk.{k}": v for k, v in state.trunk.state_dict().items()}
    for path, head in state.heads.items():
        tensors.update({f"heads.{path}.weight": head.weight,
                        f"heads.{path}.bias": head.bias,
                        f"u.{path}": state.u[path], f"v.{path}": state.v[path]})
    return tensors


def emit(state):
    """Learned bases are durable parameters, not reconstructed from the base."""
    header = json.dumps({"format": "learned_tasks_v1", "scope": state.scope,
                         "rank": state.rank, "latent": state.latent,
                         "hidden": state.hidden, "paths": list(state.heads)}, sort_keys=True).encode()
    tensors = {"posterior.mu": state.mu, "posterior.log_std": state.log_std,
               **shared_tensors(state)}
    body = st_save({key: value.detach().cpu().contiguous() for key, value in tensors.items()})
    return len(header).to_bytes(8, "big") + header + body


def load(state, payload):
    """Same scope resumes q; a new scope inherits the frozen decoder only."""
    header, tensors = unpack(payload)
    expected = ("learned_tasks_v1", state.rank, state.latent, state.hidden, list(state.heads))
    if tuple(header[k] for k in ("format", "rank", "latent", "hidden", "paths")) != expected:
        raise ValueError("checkpoint and learned decoder architecture differ")
    same_scope = header["scope"] == state.scope
    if not same_scope and state.train_decoder:
        raise ValueError("a new task scope must freeze the shared decoder")
    if same_scope and tensors["posterior.mu"].shape != state.mu.shape:
        raise ValueError("same task scope requires the same posterior bank")
    with torch.no_grad():
        for key, tensor in shared_tensors(state).items():
            tensor.copy_(tensors[key])
        if same_scope:
            state.mu.copy_(tensors["posterior.mu"])
            state.log_std.copy_(tensors["posterior.log_std"])
        else:
            state.mu.zero_()
            state.log_std.zero_()


def materialize(state, z) -> LoraState:
    """One sampled latent becomes ordinary LoRA factors with identical math."""
    if z.shape != (state.latent,):
        raise ValueError("materialize requires exactly one latent vector")
    a, b = {}, {}
    with torch.no_grad():
        for path in state.heads:
            a[path] = torch.nn.Parameter(normalize(state.v[path], dim=0).T.contiguous(), requires_grad=False)
            b[path] = torch.nn.Parameter(normalize(state.u[path], dim=0) @ cores(state, path, z)[0], requires_grad=False)
    return LoraState(state.rank, a, b)
