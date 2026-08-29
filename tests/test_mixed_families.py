"""Two adapter FAMILIES at one site path on one shared learner (the sweep's
own failure, pinned): a lora tenant and a plora tenant claim the same
Linears, and each row must be served by its own family's math.

The claims: wrappers of different families NEST (join_site), each applies
only rows whose routed state is its own and passes the rest through; a
tenant's forward beside a foreign-family tenant is BIT-IDENTICAL to the same
tenant alone (the transparent case, across the chain); install order does
not matter; and uninstalling either tenant splices its wrapper out of the
chain wherever it sits, restoring the bare Linear when the last leaves.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:                     # stdlib-only environment
    torch = None

if torch is not None:
    from rlstack.policy.adapters import lora_torch, plora_torch
    from rlstack.policy.adapters.replay import ReplayRows, SiteWrapper, row_plan
    from rlstack.policy.siteschema import SiteMeta


def tiny_linear_model():
    model = torch.nn.Module()
    model.proj = torch.nn.Linear(8, 8, bias=False)
    torch.manual_seed(0)
    with torch.no_grad():
        model.proj.weight.copy_(torch.randn(8, 8))
    return model


def lora_state():
    meta = SiteMeta(name="proj", path="proj", has_weight=True,
                    shape=(8, 8), is_boundary=False)
    state = lora_torch.build((meta,), {"r": 2, "seed": 3})
    with torch.no_grad():                       # nonzero delta, so it shows
        state.b["proj"].copy_(torch.randn(
            8, 2, generator=torch.Generator().manual_seed(7)) * 0.1)
    return state


def plora_state(model):
    meta = SiteMeta(name="proj", path="proj", has_weight=True,
                    shape=(8, 8), is_boundary=False)
    state = plora_torch.build(
        (meta,), {"k": 2, "latent": 4, "members": 2, "prior_std": 0.05,
                  "hidden": 8, "factors": "cas://unused", "seed": 5})
    with torch.no_grad():                       # nonzero head, so it shows
        state.heads["proj"].copy_(torch.randn(
            4, 8, generator=torch.Generator().manual_seed(9)) * 0.1)
    return state


def routed(model, slot, facts=None):
    return row_plan(model).route(ReplayRows(
        slots=(slot,), index=torch.zeros(2, dtype=torch.long), facts=facts))


PLORA_FACTS = (({"plora_eps": [0.1, -0.2, 0.3, 0.4]},),
               ({"plora_eps": [0.1, -0.2, 0.3, 0.4]},))


@unittest.skipIf(torch is None, "torch-gated")
class MixedFamilyTest(unittest.TestCase):
    def forward(self, model, slot, facts=None):
        x = torch.linspace(-1.0, 1.0, 32).reshape(2, 2, 8)
        with routed(model, slot, facts):
            return model.proj(x)

    def test_each_family_is_untouched_by_the_other(self) -> None:
        """The sweep's crash, as arithmetic: lora rows through a chain that
        also holds plora (and vice versa) equal each family alone."""
        for order in ("lora-first", "plora-first"):
            model = tiny_linear_model()
            lora = lora_state()
            plora = plora_state(model)
            alone = tiny_linear_model()
            if order == "lora-first":
                lora_torch.install(model, lora)
                plora_torch.install(model, plora)
            else:
                plora_torch.install(model, plora)
                lora_torch.install(model, lora)

            lora_mine = lora_state()
            lora_torch.install(alone, lora_mine)
            self.assertTrue(torch.equal(
                self.forward(model, {"proj": lora}),
                self.forward(alone, {"proj": lora_mine})), order)

            plora_alone = tiny_linear_model()
            plora_mine = plora_state(plora_alone)
            plora_torch.install(plora_alone, plora_mine)
            self.assertTrue(torch.equal(
                self.forward(model, {"proj": plora}, PLORA_FACTS),
                self.forward(plora_alone, {"proj": plora_mine},
                             PLORA_FACTS)), order)

    def test_the_chain_nests_and_unsplices_in_any_order(self) -> None:
        model = tiny_linear_model()
        bare = model.proj
        lora, plora = lora_state(), plora_state(model)
        lora_torch.install(model, lora)
        plora_torch.install(model, plora)
        self.assertIsInstance(model.proj, SiteWrapper)
        self.assertIsInstance(model.proj.inner, SiteWrapper)

        lora_torch.uninstall(model, lora)       # the INNER-or-OUTER one leaves
        self.assertIsInstance(model.proj, SiteWrapper)
        self.assertIs(model.proj.inner, bare)
        plora_torch.uninstall(model, plora)
        self.assertIs(model.proj, bare)

    def test_a_second_same_family_tenant_joins_through_the_chain(self) -> None:
        """A lora tenant arriving after plora wrapped the head must JOIN the
        nested LoraSite, never stack a second one."""
        model = tiny_linear_model()
        first, second = lora_state(), lora_state()
        plora = plora_state(model)
        lora_torch.install(model, first)
        plora_torch.install(model, plora)       # wraps around
        lora_torch.install(model, second)       # joins the nested LoraSite
        wrappers = []
        node = model.proj
        while isinstance(node, SiteWrapper):
            wrappers.append(type(node).__name__)
            node = node.inner
        self.assertEqual(sorted(wrappers), ["LoraSite", "PloraSite"])


if __name__ == "__main__":
    unittest.main()
