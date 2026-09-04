"""I1 as an import invariant: the membrane is enforced by the module graph.

The folder layout mirrors the architecture, so the architecture is checkable:
the two worlds never import each other, and the data layer (the membrane)
imports no other rlstack package. Type-only imports count too — even static
coupling across the membrane is a smell.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "rlstack"

HEAVY_REGIONS = ("runner/engines", "runner/learners", "runner/transports")
"""STYLE rule 8's three heavy FOLDERS: real inference metal, real training
metal, real wire substrates. STYLE.md's tree names the first two;
`runner/transports/` is ADR 0007's new region and the line naming it there is
Samarth's edit to make — this table is the enforcement either way."""

TORCH_SEAMS = ("policy/adapters/replay.py", "policy/adapters/plora_factors.py")
"""The two heavy files under `policy/adapters/` that carry no side suffix:
the replay seam (#44) and plora's factor math. Both are torch, both are
reached only from a training process."""

HEAVY_SUBSTRATES = ("torch", "vllm", "modal")
"""The dependencies the heavy files exist to contain. `import rlstack` in a
stdlib-only interpreter must not reach any of them — which the local fakes
suite proves every run, and this test names."""


def loads_lazily(relative: str) -> bool:
    """Is this file one the rule is ABOUT — a heavy region's, or one of an
    adapter type's PER-SIDE lowerings? Those import their substrate at module
    scope BY DESIGN (STYLE rule 8: `runner/engines/`, `runner/learners/`,
    `runner/transports/`, and `policy/adapters/` one file per adapter type per
    side, plus the two torch seams beside them), and every one of them is
    reached only from a function assembling a GPU run or wiring a venue door.
    Everything else is what the rule guards."""
    if any(relative.startswith(region) for region in HEAVY_REGIONS):
        return True
    return (relative.startswith("policy/adapters/")
            and (relative.endswith(("_torch.py", "_vllm.py"))
                 or relative in TORCH_SEAMS))


def rlstack_imports(path: Path) -> set[str]:
    """Every `rlstack.*` module this file imports (TYPE_CHECKING included)."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out |= {alias.name for alias in node.names
                    if alias.name.startswith("rlstack")}
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("rlstack"):
                out.add(node.module)
    return out


def imports_by_region(region: str) -> dict[str, set[str]]:
    """{relative file -> its rlstack imports} for one subpackage."""
    return {
        str(path.relative_to(PACKAGE)): rlstack_imports(path)
        for path in (PACKAGE / region).rglob("*.py")
    }


class TestMembrane(unittest.TestCase):
    def assert_never_imports(self, region: str, forbidden: str) -> None:
        for file, imports in imports_by_region(region).items():
            bad = {i for i in imports if i.startswith(forbidden)}
            self.assertFalse(bad, f"{file} imports {sorted(bad)} — across the membrane")

    def test_training_never_imports_inference(self) -> None:
        self.assert_never_imports("training", "rlstack.inference")

    def test_inference_never_imports_training(self) -> None:
        self.assert_never_imports("inference", "rlstack.training")

    def test_observe_reads_the_membrane_only(self) -> None:
        """The observer derives views from store bytes alone: it may import
        the data layer (stores) and nothing else — no specs, no registries,
        no runner. Its dictionary comes from the run dir, never re-derived."""
        for file, imports in imports_by_region("observe").items():
            bad = {i for i in imports
                   if not i.startswith(("rlstack.data", "rlstack.observe"))}
            self.assertFalse(bad, f"{file} imports {sorted(bad)} — the "
                                  f"observer reads committed bytes only")

    def test_observe_web_is_the_only_home_of_static_assets(self) -> None:
        """rule 8: observe/web/ holds the UI as real files (index.html,
        style.css, native ES modules) and is the package's ONE folder of
        non-.py files — no build step, no CDN, nothing generated. Everywhere
        else under rlstack/ is Python."""
        web = PACKAGE / "observe" / "web"
        self.assertTrue((web / "index.html").is_file(), "the document must ship")
        stray = sorted(str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*")
                       if path.is_file() and path.suffix not in (".py", ".pyc")
                       and web not in path.parents)
        self.assertFalse(stray, f"non-python files outside observe/web/: {stray}")

    def test_data_imports_no_other_rlstack_package(self) -> None:
        """The membrane is dumb: no specs, no registries, no worlds."""
        for file, imports in imports_by_region("data").items():
            bad = {i for i in imports if not i.startswith("rlstack.data")}
            self.assertFalse(bad, f"{file} imports {sorted(bad)} — data/ must stay dumb")

    def test_spec_values_import_nothing_from_rlstack(self) -> None:
        """Specs and identity are pure values; only validate joins the registries."""
        for module in ("spec/specs.py", "spec/canonical.py"):
            imports = rlstack_imports(PACKAGE / module)
            self.assertFalse(imports, f"{module} imports {sorted(imports)}")

    def test_the_learners_import_no_spec_class(self) -> None:
        """ADR 0002's acid test, in #69's shape: a learner owns tensors, not
        experiments. What reaches it is a Parameterization built by the runner
        — the loss and every adapter type BY REGISTRY KEY, seeds derived,
        sites resolved — so runner/learners/ has no business importing
        rlstack.spec."""
        for file, imports in imports_by_region("runner/learners").items():
            bad = {i for i in imports if i.startswith("rlstack.spec")}
            self.assertFalse(bad, f"{file} imports {sorted(bad)} — a learner "
                                  f"may not read the experiment")

    def test_the_heavy_regions_are_imported_lazily(self) -> None:
        """STYLE RULE 7, PINNED (ADR 0007, Q10). vLLM, torch and a venue SDK
        sit at MODULE scope in the files whose region exists to hold them and
        nowhere else — so no other module under rlstack/ may import those
        substrates, OR those files, at module scope. Everywhere else they
        belong inside the function that is assembling a GPU run or wiring a
        venue door.

        The discipline held by hand until now; a third heavy region
        (`runner/transports/`, whose one stray import would make
        `import rlstack` require modal and kill the stdlib-only fakes suite)
        is the moment to make it a test."""
        for path in sorted(PACKAGE.rglob("*.py")):
            relative = str(path.relative_to(PACKAGE))
            if loads_lazily(relative):
                continue
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                self.assert_no_heavy_import(relative, node)

    def assert_no_heavy_import(self, relative: str, node: ast.stmt) -> None:
        """One module-scope statement, checked: an `import X` or a
        `from X import ...` naming a heavy substrate, or naming a file that
        loads lazily, is the violation. Only a module's `body` is walked — an
        import nested in a function is exactly the lazy import the rule asks
        for."""
        named: set[str] = set()
        if isinstance(node, ast.Import):
            named = {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            named = {node.module}
        for name in named:
            self.assertNotIn(
                name.split(".")[0], HEAVY_SUBSTRATES,
                f"{relative} imports {name!r} at module scope — a heavy "
                f"substrate belongs in its own file, imported lazily")
            if not name.startswith("rlstack."):
                continue
            target = name[len("rlstack."):].replace(".", "/") + ".py"
            self.assertFalse(
                loads_lazily(target),
                f"{relative} imports {name!r} at module scope — the heavy "
                f"files load lazily, from the function that needs them")

    def test_the_client_library_never_imports_engine_code(self) -> None:
        """rlstack_engine ships in the ENGINE image; rlstack names plugins by
        string only. The import is one-way: engine code may import rlstack
        types, the client library must never import back."""
        for path in PACKAGE.rglob("*.py"):
            imports = rlstack_imports(path)
            bad = {i for i in imports if i.startswith("rlstack_engine")}
            self.assertFalse(
                bad, f"{path.relative_to(PACKAGE)} imports {sorted(bad)}")


if __name__ == "__main__":
    unittest.main()
