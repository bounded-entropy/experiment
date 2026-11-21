"""The chunked replay calls a sharded model's trunk and head, never its root.

FsdpTorchLearner wraps each decoder block and then the root with FSDP2's
`fully_shard`; the root's group holds what no block owns (the embedding, the
final norm, the head, tied or not). The replay (`trunk_and_head`) calls
`model.model` and then the head chunk by chunk, so the root's forward hooks
never run and the embedding meets a DTensor weight. `unshard_the_root` gathers
the root's group first. This suite reproduces the failure on a one-rank gloo
group on CPU, then checks that with the root gathered the forward and the
backward run, give the unwrapped model's gradient, and that FSDP reshards the
root at the end of the backward so the next microbatch starts sharded again.
Torch-gated, and run in a child process so the process group never leaks into
another test.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest

try:
    import torch
    from torch.distributed.fsdp import fully_shard  # noqa: F401
except ImportError:          # the stdlib-only fakes suite, or a torch without FSDP2
    torch = None

CHILD = textwrap.dedent(r"""
    import copy, os, socket, types
    import torch, torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor
    from rlstack.runner.learners.fsdp_torch import unshard_the_root
    from rlstack.runner.learners.torch_learner import head_logprobs, trunk_and_head

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", (1,))
    torch.manual_seed(0)
    V, D = 37, 8
    DELTA = [None]

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.mlp = torch.nn.Linear(D, D)
        def forward(self, h):
            # the deltas live inside the blocks, unregistered, as a routed LoRA does
            return h + torch.tanh(self.mlp(h)) * DELTA[0]

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(V, D)
            self.layers = torch.nn.ModuleList([Block(), Block()])
            self.norm = torch.nn.LayerNorm(D)
        def forward(self, input_ids, attention_mask=None, use_cache=False):
            h = self.embed_tokens(input_ids)
            for layer in self.layers:
                h = layer(h)
            return types.SimpleNamespace(last_hidden_state=self.norm(h))

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Trunk()
            self.lm_head = torch.nn.Linear(D, V, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight        # tied, as Qwen's small models are
        def get_output_embeddings(self):
            return self.lm_head
        def forward(self, input_ids, **_):
            return types.SimpleNamespace(logits=self.lm_head(self.model(input_ids).last_hidden_state))

    plain = LM().double().requires_grad_(False)
    sharded = copy.deepcopy(plain)
    for block in sharded.model.layers:
        fully_shard(block, mesh=mesh, reshard_after_forward=True)
    fully_shard(sharded, mesh=mesh)
    delta = torch.nn.Parameter(torch.full((D,), 1.1, dtype=torch.float64))   # the trainable thing, whole
    DELTA[0] = delta
    ids = torch.tensor([[1, 5, 9, 2, 30, 4]])

    def loss_of(model):
        trunk, head = trunk_and_head(model)
        hidden = trunk(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).last_hidden_state
        return -head_logprobs(hidden[:, :-1], head, ids[:, 1:], chunk=2).sum()

    try:
        loss_of(sharded)
        raise SystemExit("the bypass did not fail: the reproduction is stale")
    except RuntimeError as error:
        assert "DTensor" in str(error), error

    reference = loss_of(plain)
    reference.backward()
    want, delta.grad = delta.grad.clone(), None
    for step in range(2):                       # two microbatches: the second starts resharded
        unshard_the_root(sharded)
        assert not isinstance(sharded.model.embed_tokens.weight, DTensor)
        got = loss_of(sharded)
        got.backward()
        assert torch.allclose(got, reference), (got, reference)
        assert torch.allclose(delta.grad, want * (step + 1)), (delta.grad, want)
        assert isinstance(sharded.model.embed_tokens.weight, DTensor), "the root was not resharded"
    dist.destroy_process_group()
    print("ok")
""")


@unittest.skipIf(torch is None, "torch with FSDP2 is not installed")
class UnshardTheRootTest(unittest.TestCase):

    def test_the_chunked_replay_runs_on_a_sharded_model_once_the_root_is_gathered(self) -> None:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))}
        done = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                              timeout=300)
        self.assertEqual(done.returncode, 0, done.stderr[-3000:] + done.stdout[-1000:])
        self.assertEqual(done.stdout.strip().splitlines()[-1], "ok")


if __name__ == "__main__":
    unittest.main()
