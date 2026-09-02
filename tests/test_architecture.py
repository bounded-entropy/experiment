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
