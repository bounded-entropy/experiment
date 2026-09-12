"""A Gaussian task bank over the existing spectral gain map.

Reuse spectral_latent's state, hypernetwork, Gaussian KL and factorization.
Support is explicitly the leading k singular directions. Only checkpoint
ownership and posterior routing differ from the single-task adapter.
"""

from dataclasses import dataclass, field, replace
import json

import torch
from safetensors.torch import save as st_save

from rlstack.policy.adapters import spectral_latent_torch as spectral
from rlstack.policy.adapters.plora_torch import analytic_kl
from rlstack.policy.adapters.replay import SiteWrapper, join_site, leave_site, leaf_module
from rlstack.policy.adapters.spectral_tasks import EPS_RECORD, TASK_RECORD
from rlstack.policy.adapters.spectral_torch import FROZEN_DTYPE, full_spectrum


@dataclass
class TaskState:
    decoder: spectral.SlatentState
    scope: str
    train_decoder: bool
    scales: str
    beta: float
    active_tasks: torch.Tensor | None = field(default=None, repr=False)

    def parameters(self):
        return [p for group in param_groups(self).values() for p in group]


def build(sites, init):
    """Independent prior posteriors and one shared zero-gain decoder."""
    k, tasks = int(init["k"]), int(init["tasks"])
    if tasks < 1 or any(s.shape is None or k > min(s.shape) for s in sites):
        raise ValueError("positive task count and k fitting every matrix required")
    scales = init.get("scales", "learned")
    if scales not in ("learned", "fixed", "deterministic"):
        raise ValueError(f"unknown posterior scales: {scales}")
    if init.get("prior", "fixed") != "fixed" or init.get("prior_std", 1.0) != 1.0:
        raise ValueError("spectral_tasks uses the fixed N(0,I) prior")
    # Head width is k; install uses the actual base matrix shapes.
    decoder = spectral.build(tuple(replace(s, shape=(k, k)) for s in sites),
                             {**init, "members": 1, "prior_std": 1.0})
    decoder.mu = torch.nn.Parameter(torch.zeros(tasks, decoder.latent))
    decoder.log_std = torch.nn.Parameter(torch.zeros(tasks, decoder.latent),
                                         requires_grad=scales == "learned")
    train_decoder = bool(init.get("train_decoder", True))
    for p in decoder.mapper():
        p.requires_grad_(train_decoder)
    return TaskState(decoder, str(init["scope"]), train_decoder, scales,
                     float(init.get("beta", 1e-3)))


def param_groups(state):
    decoder = state.decoder
    groups = {"mean": [decoder.mu]}
    if state.scales == "learned":
        groups["scale"] = [decoder.log_std]
    if state.train_decoder:
        groups["mapper"] = decoder.mapper()
    return groups


def task_latents(state, rows):
    """One recorded draw per sequence, fixed throughout its tokens."""
    if rows.facts is None:
        raise ValueError("spectral_tasks requires recorded task indices and noise")
    tasks, noise = [], []
    decoder = state.decoder
    for turns in rows.facts:
        ids = {int(turn[TASK_RECORD]) for turn in turns}
        draws = {tuple(turn[EPS_RECORD]) for turn in turns}
        if len(ids) != 1 or len(draws) != 1:
            raise ValueError("one task and one latent draw per sequence required")
        task, eps = ids.pop(), draws.pop()
        if not 0 <= task < decoder.mu.shape[0] or len(eps) != decoder.latent:
            raise ValueError("task index or latent width does not fit this bank")
        tasks.append(task)
        noise.append(eps)
    indices = torch.tensor(tasks, device=decoder.mu.device)
    state.active_tasks = indices
    mu = decoder.mu[indices]
    if state.scales == "deterministic":
        return mu
    eps = torch.tensor(noise, dtype=mu.dtype, device=mu.device)
    return mu + decoder.log_std[indices].exp() * eps


class TaskSite(SiteWrapper):
    def forward(self, x):
        rows = self.plan.rows
        slot = rows.uniform()
        if slot is None:
            raise ValueError("one tenant per forward required")
        state = slot.get(self.path)
        if not isinstance(state, TaskState):
            return self.inner(x)
        decoder = state.decoder
        z = task_latents(state, rows)
        if x.dim() != 3 or x.shape[0] != z.shape[0]:
            raise ValueError("task posteriors require [sequences,tokens,width]")
        eff = spectral.effective_gains(decoder, self.path, z)
        decoder.served.setdefault(self.path, []).append(eff)
        v, u = decoder.v[self.path].float(), decoder.u[self.path].float()
        delta = ((x.float() @ v) * eff[:, None, :]) @ u.T
        return self.inner(x) + delta.to(x.dtype)


def provide(state):
    decoder = state.decoder
    ids = state.active_tasks
    if ids is None:
        ids = torch.arange(decoder.mu.shape[0], device=decoder.mu.device)
    mu, log_std = decoder.mu[ids], decoder.log_std[ids]
    # For a deterministic latent this is mean shrinkage, not finite KL.
    penalty = (0.5 * mu.square().sum() if state.scales == "deterministic"
               else analytic_kl(mu, log_std, decoder.prior_log_std)) / len(ids)
    return {"latent_kl": penalty,
            "spectral_sigma_mean": (log_std.exp().mean() if state.scales != "deterministic"
                                     else mu.new_tensor(0.0)),
            "spectral_prior_std": mu.new_tensor(1.0),
            "factual_beta": mu.new_tensor(state.beta),
            **spectral.served_summary(decoder)}


def install(model, state):
    decoder = state.decoder
    for path in decoder.paths:
        parent, leaf = leaf_module(model, path)
        node = getattr(parent, leaf)
        while isinstance(node, SiteWrapper):
            node = node.inner
        weight = node.weight
        u, sigma, v = full_spectrum(weight)
        decoder.u[path] = u[:, :decoder.k].to(weight.device, FROZEN_DTYPE)
        decoder.v[path] = v[:, :decoder.k].to(weight.device, FROZEN_DTYPE)
        decoder.sigma[path] = sigma[:decoder.k].to(weight.device)
        for p in (*decoder.parameters(), *decoder.mapper(), decoder.prior_log_std):
            p.data = p.data.to(weight.device)
        join_site(model, path, TaskSite, state)


def uninstall(model, state):
    for path in state.decoder.paths:
        leave_site(model, path, TaskSite, state)


def emit(state):
    """Checkpoint the decoder and posteriors; rebuild the fixed basis."""
    d = state.decoder
    tensors = {"posterior.mu": d.mu.detach().cpu(),
               "posterior.log_std": d.log_std.detach().cpu()}
    tensors.update({f"trunk.{k}": v.detach().cpu() for k, v in d.trunk.state_dict().items()})
    tensors.update({f"heads.{k}": v.detach().cpu() for k, v in d.heads.items()})
    head = json.dumps({"scope": state.scope, "k": d.k, "latent": d.latent,
                       "hidden": d.hidden, "paths": d.paths, "bound": d.bound,
                       "support": "leading"}, sort_keys=True).encode()
    return len(head).to_bytes(8, "big") + head + st_save(tensors)


def load(state, payload):
    """Same scope restores q; a new scope inherits only the frozen decoder."""
    head, tensors = spectral.unpack(payload)
    d = state.decoder
    expected = (d.k, d.latent, d.hidden, list(d.paths), d.bound, "leading")
    actual = tuple(head[k] for k in ("k", "latent", "hidden", "paths", "bound", "support"))
    if actual != expected:
        raise ValueError("checkpoint and declared spectral decoder differ")
    d.trunk.load_state_dict({k[6:]: v for k, v in tensors.items() if k.startswith("trunk.")})
    for path, p in d.heads.items():
        p.data.copy_(tensors[f"heads.{path}"])
    if head["scope"] == state.scope:
        if tensors["posterior.mu"].shape != d.mu.shape:
            raise ValueError("same task scope requires the same posterior bank")
        d.mu.data.copy_(tensors["posterior.mu"])
        d.log_std.data.copy_(tensors["posterior.log_std"])
    elif state.train_decoder:
        raise ValueError("new task scopes inherit a frozen decoder only")
    else:
        d.mu.data.zero_()
        d.log_std.data.zero_()
