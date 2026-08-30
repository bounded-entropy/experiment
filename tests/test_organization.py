"""The organization plane (#58): folders, annotations, and substring search.

Three claims, each with its own class.

FOLDERS ARE STORE ROOTS, CHOSEN AT BIRTH. A run's organizational location is
the directory its store was constructed with; the observer takes a TOP
directory and discovers the roots beneath it, never descending into one. A top
that is itself a root is the degenerate single-store case — the deployed UI's
/store mount — and everything about it must read exactly as it did before.

ANNOTATIONS ARE FLAVORTEXT. annotations.jsonl sits beside runs/, is append
only, merges latest-wins PER FIELD, is never hashed, and is read by nobody but
a human. tests/test_resume.py holds the other half of that claim: annotating a
run changes not one byte of its run directory.

SEARCH IS A SUBSTRING. Case-insensitive, over name + tags + note + run_id, and
nothing fancier — stated once in views.matches and mirrored by the page.
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rlstack import LocalStore
from rlstack.__main__ import main
from rlstack.data.stores.base import StoreError
from rlstack.observe.locate import Root, roots_for, roots_under, rooted
from rlstack.observe.page import asset
from rlstack.observe.ui import ui_app
from rlstack.observe.views import matches, render_runs, runs_data
from test_ui import call

MANIFEST = {"run_id": "abc123", "spec": json.dumps(
    {"algo": {"schedule": {"n_updates": 2}}})}


def seeded(root: Path, run_id: str = "abc123", host: str = "l4-a") -> LocalStore:
    """A store root holding one run and one host journal that placed it."""
    store = LocalStore(root)
    store.open_run(run_id, dict(MANIFEST, run_id=run_id))
    store.append_host_event(host, {"event": "attach", "t": 10.0,
                                   "run_id": run_id, "pools": ["policy"],
                                   "n_updates": 2, "store": store.describe()})
    return store


class AnnotationVerbTest(unittest.TestCase):
    """The three store verbs: append one row, merge latest-wins per field,
    and live BESIDE runs/ rather than inside it."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = LocalStore(self.root)

    def test_the_file_sits_beside_runs_never_inside_one(self) -> None:
        self.assertEqual(self.store.annotations_key(), "annotations.jsonl")
        self.store.annotate_run("abc123", name="the good one")
        path = self.store.path_of(self.store.annotations_key())
        self.assertEqual(path.parent, self.root)
        self.assertFalse((self.root / "runs").exists())   # no run dir touched

    def test_a_row_carries_only_the_fields_passed(self) -> None:
        self.store.annotate_run("abc123", name="lr sweep 3")
        row = json.loads(self.store.path_of("annotations.jsonl").read_text())
        self.assertEqual(set(row), {"t", "run_id", "name"})
        self.assertEqual(row["run_id"], "abc123")
        self.assertIsInstance(row["t"], float)

    def test_the_merge_is_latest_wins_per_field(self) -> None:
        self.store.annotate_run("abc123", name="first", tags=["math", "l3"],
                                note="started from scratch")
        self.store.annotate_run("abc123", note="the teacher was 32B")
        self.store.annotate_run("abc123", tags=["math"])
        self.store.annotate_run("other", name="unrelated")
        merged = self.store.read_annotations()
        self.assertEqual(merged["abc123"]["name"], "first")     # never restated
        self.assertEqual(merged["abc123"]["note"], "the teacher was 32B")
        self.assertEqual(merged["abc123"]["tags"], ["math"])    # the WHOLE list
        self.assertEqual(merged["other"], {"name": "unrelated"})

    def test_append_only_keeps_every_row(self) -> None:
        for name in ("a", "b", "c"):
            self.store.annotate_run("abc123", name=name)
        rows = self.store.path_of("annotations.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 3)
        self.assertEqual(self.store.read_annotations()["abc123"]["name"], "c")

    def test_a_torn_line_is_skipped_not_repaired(self) -> None:
        self.store.annotate_run("abc123", name="good")
        with open(self.store.path_of("annotations.jsonl"), "a") as handle:
            handle.write('{"run_id": "abc12')
        self.assertEqual(self.store.read_annotations()["abc123"]["name"], "good")

    def test_annotating_nothing_raises(self) -> None:
        with self.assertRaises(StoreError):
            self.store.annotate_run("abc123")

    def test_an_unannotated_store_reads_empty(self) -> None:
        self.assertEqual(self.store.read_annotations(), {})


class RootDiscoveryTest(unittest.TestCase):
    """A directory holding runs/, hosts/, fleet/ or annotations.jsonl IS a
    store root; discovery stops there, because everything below it is that
    store's own key tree."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(tmp.name)

    def test_the_degenerate_case_is_one_root_named_empty(self) -> None:
        """A top that is itself a store — the deployed /store mount — must
        read exactly as it did before folders existed."""
        seeded(self.top)
        roots = roots_under(str(self.top))
        self.assertEqual([root.folder for root in roots], [""])
        self.assertEqual(roots[0].store.describe(), str(self.top))
        self.assertEqual(roots[0].store.list_runs(), ["abc123"])

    def test_an_empty_top_is_the_degenerate_case_too(self) -> None:
        """A directory a store is about to be born in is not two folders."""
        roots = roots_under(str(self.top))
        self.assertEqual([root.folder for root in roots], [""])

    def test_nested_roots_are_found_with_relative_folders(self) -> None:
        seeded(self.top / "opd" / "math" / "l3-5")
        seeded(self.top / "sft" / "base")
        (self.top / "scratch" / "empty").mkdir(parents=True)
        self.assertEqual([root.folder for root in roots_under(str(self.top))],
                         ["opd/math/l3-5", "sft/base"])

    def test_discovery_never_descends_into_a_root(self) -> None:
        """runs/ inside a root is the key tree, not another folder — and a
        directory that happens to sit beside it is not one either."""
        store = seeded(self.top / "a")
        store.open_run("deep", dict(MANIFEST, run_id="deep"))
        (self.top / "a" / "b" / "runs").mkdir(parents=True)
        self.assertEqual([root.folder for root in roots_under(str(self.top))],
                         ["a"])

    def test_a_root_is_any_part_of_the_key_tree(self) -> None:
        for mark in ("runs", "hosts", "fleet"):
            (self.top / mark / "x").mkdir(parents=True)
            self.assertEqual([r.folder for r in roots_under(str(self.top))], [""])
            for path in sorted((self.top / mark).rglob("*"), reverse=True):
                path.rmdir()
            (self.top / mark).rmdir()
        LocalStore(self.top / "tagged").annotate_run("abc123", name="only a name")
        self.assertEqual([r.folder for r in roots_under(str(self.top))], ["tagged"])

    def test_file_urls_and_hidden_directories(self) -> None:
        seeded(self.top / "kept")
        (self.top / ".hidden" / "runs").mkdir(parents=True)
        self.assertEqual([r.folder for r in roots_under("file://" + str(self.top))],
                         ["kept"])

    def test_a_bare_store_reads_as_the_degenerate_root(self) -> None:
        store = seeded(self.top)
        self.assertEqual(rooted([store]), [Root("", store)])

    def test_several_tops_are_prefixed_by_their_own_names(self) -> None:
        seeded(self.top / "left" / "opd")
        seeded(self.top / "right")
        roots = roots_for([str(self.top / "left"), str(self.top / "right")])
        self.assertEqual([root.folder for root in roots], ["left/opd", "right"])

    def test_two_tops_with_the_same_name_are_refused(self) -> None:
        """A folder addresses one root; two roots on one address is a link
        that means two things."""
        for side in ("a", "b"):
            seeded(self.top / side / "store")
        with self.assertRaises(ValueError):
            roots_for([str(self.top / "a" / "store"),
                       str(self.top / "b" / "store")])


class RunsViewTest(unittest.TestCase):
    """The runs view over roots: one row per (folder, run_id), each carrying
    the annotations its OWN root holds."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(tmp.name)
        self.left = seeded(self.top / "opd" / "math", run_id="abc123")
        self.right = seeded(self.top / "sft", run_id="abc123", host="l4-b")
        self.left.annotate_run("abc123", name="teacher 32B",
                               tags=["opd", "l3-5"], note="kl fell")
        self.roots = roots_under(str(self.top))

    def test_the_same_id_in_two_folders_is_two_rows(self) -> None:
        rows = runs_data(self.roots)
        self.assertEqual(sorted(r["folder"] for r in rows), ["opd/math", "sft"])
        self.assertEqual({r["run_id"] for r in rows}, {"abc123"})
        self.assertTrue(all(r["forked"] for r in rows))
        self.assertEqual(sorted(rows[0]["in_folders"]), ["opd/math", "sft"])

    def test_annotations_ride_with_their_own_root(self) -> None:
        rows = {r["folder"]: r for r in runs_data(self.roots)}
        self.assertEqual(rows["opd/math"]["name"], "teacher 32B")
        self.assertEqual(rows["opd/math"]["tags"], ["opd", "l3-5"])
        self.assertEqual(rows["opd/math"]["note"], "kl fell")
        self.assertEqual(rows["sft"]["name"], "")       # a different folder,
        self.assertEqual(rows["sft"]["tags"], [])       # a different run

    def test_search_is_a_case_insensitive_substring(self) -> None:
        row = {r["folder"]: r for r in runs_data(self.roots)}["opd/math"]
        for needle in ("", "TEACHER", "l3-5", "kl fell", "abc1"):
            self.assertTrue(matches(row, needle), needle)
        for needle in ("gspo", "sft", "zzz"):
            self.assertFalse(matches(row, needle), needle)

    def test_the_terminal_view_groups_by_folder_and_greps(self) -> None:
        text = render_runs(self.roots)
        self.assertIn("folder opd/math", text)
        self.assertIn("folder sft", text)
        self.assertIn("teacher 32B", text)
        self.assertIn("opd,l3-5", text)
        filtered = render_runs(self.roots, grep="teacher")
        self.assertIn("folder opd/math", filtered)
        self.assertNotIn("folder sft", filtered)
        self.assertIn("no experiment matching", render_runs(self.roots, grep="zzz"))

    def test_the_degenerate_view_has_no_folder_header(self) -> None:
        """One root, folder "": the table reads as it always has."""
        text = render_runs([self.left])
        self.assertNotIn("folder ", text)
        self.assertIn("abc123", text)


class AddressingTest(unittest.TestCase):
    """Every link carries its folder, and a bare id that names two runs is
    answered with the AMBIGUITY — never with a silent pick."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(tmp.name)
        seeded(self.top / "opd" / "math", run_id="abc123")
        seeded(self.top / "sft", run_id="abc123", host="l4-b")
        seeded(self.top / "sft", run_id="only-here", host="l4-b")
        self.app = ui_app(roots_under(str(self.top)))

    def test_a_bare_id_in_two_folders_answers_with_the_folders(self) -> None:
        status, _, body = call(self.app, "/api/run/abc123")
        self.assertEqual(status, "200 OK")
        payload = json.loads(body)
        self.assertEqual(sorted(payload["ambiguous"]), ["opd/math", "sft"])
        self.assertEqual(payload["run_id"], "abc123")

    def test_every_run_route_says_so_not_just_the_first(self) -> None:
        for tail in ("", "/timing", "/waves", "/wave/1"):
            _, _, body = call(self.app, f"/api/run/abc123{tail}")
            self.assertIn("ambiguous", json.loads(body), tail)

    def test_the_folder_resolves_it(self) -> None:
        status, _, body = call(self.app, "/api/run/abc123?root=opd/math")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body)["run_id"], "abc123")
        self.assertNotIn("ambiguous", json.loads(body))

    def test_an_unknown_folder_is_a_404(self) -> None:
        status, _, _ = call(self.app, "/api/run/abc123?root=nowhere")
        self.assertEqual(status, "404 Not Found")

    def test_a_unique_id_needs_no_folder(self) -> None:
        status, _, body = call(self.app, "/api/run/only-here")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body)["run_id"], "only-here")

    def test_the_index_carries_the_folder_and_the_annotation(self) -> None:
        _, _, body = call(self.app, "/api/runs")
        body = __import__("json").dumps(
            __import__("json").loads(body)["runs"]).encode()
        rows = json.loads(body)
        self.assertEqual(sorted(r["folder"] for r in rows),
                         ["opd/math", "sft", "sft"])
        for row in rows:
            for field in ("folder", "name", "tags", "note"):
                self.assertIn(field, row)

    def test_the_fleet_spans_every_root_and_names_the_folders(self) -> None:
        _, _, body = call(self.app, "/api/hosts")
        fleet = json.loads(body)
        self.assertEqual(sorted(h["folder"] for h in fleet["hosts"]),
                         ["opd/math", "sft"])
        self.assertEqual(sorted(fleet["folders"]), ["opd/math", "sft"])

    def test_a_host_route_can_be_pinned_to_one_folder(self) -> None:
        status, _, body = call(self.app, "/api/host/l4-b?root=sft")
        self.assertEqual(status, "200 OK")
        self.assertEqual(json.loads(body)["folders"], ["sft"])
        status, _, _ = call(self.app, "/api/host/l4-b?root=opd/math")
        self.assertEqual(status, "404 Not Found")

    def test_the_observer_stays_read_only(self) -> None:
        """No POST route, and no route writes: the plane the UI renders is
        written by the CLI alone."""
        source = Path(__file__).resolve().parent.parent / "rlstack/observe"
        for path in sorted(source.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("annotate_run", text, f"{path.name} writes")
        for module in ("runs.js", "run.js", "nav.js"):
            page = asset(module)[0].decode("utf-8")
            self.assertNotIn("method: \"POST\"", page)
            self.assertNotIn("fetch(\"/api/annotate", page)


class PageAddressingTest(unittest.TestCase):
    """The page's half of the rule: the modules build folder-carrying links
    and render the ambiguity rather than choosing."""

    def test_the_modules_carry_the_addressing_machinery(self) -> None:
        for module, machinery in (
                ("nav.js", ("?root=", "drawAmbiguity", "apiRun", "optgroup")),
                ("runs.js", ("folderBlock", "matchExpr", "tag", "collapsed")),
                ("run.js", ("drawAmbiguity", "route.folder")),
                ("wave.js", ("drawAmbiguity", "route.folder")),
                ("host.js", ("query(route.folder)",)),
                ("fleet.js", ("shared.has(h.host)",)),
        ):
            source = asset(module)[0].decode("utf-8")
            for claim in machinery:
                self.assertIn(claim, source, f"{module} lost {claim!r}")

    def test_the_search_box_is_built_once_so_a_poll_cannot_clear_it(self) -> None:
        source = asset("runs.js")[0].decode("utf-8")
        self.assertIn('if (document.getElementById("tree")) return;', source)


class CliTest(unittest.TestCase):
    """`python -m rlstack tag <store-root> <run_id> ...` — the one writing
    verb, and the views that read it back."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.top = Path(tmp.name)
        self.store = seeded(self.top / "opd" / "math", run_id="abc123")

    def run_cli(self, *argv: str) -> str:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main(list(argv))
        return out.getvalue()

    def test_tag_round_trips_through_the_store(self) -> None:
        text = self.run_cli("tag", str(self.top / "opd" / "math"), "abc123",
                            "--name", "teacher 32B", "--tag", "opd",
                            "--tag", "l3-5", "--note", "kl fell")
        self.assertIn("teacher 32B", text)
        self.assertIn("opd, l3-5", text)
        merged = LocalStore(self.top / "opd" / "math").read_annotations()
        self.assertEqual(merged["abc123"],
                         {"name": "teacher 32B", "tags": ["opd", "l3-5"],
                          "note": "kl fell"})

    def test_a_second_tag_supersedes_only_what_it_names(self) -> None:
        self.run_cli("tag", str(self.top / "opd" / "math"), "abc123",
                     "--name", "first", "--tag", "opd")
        self.run_cli("tag", str(self.top / "opd" / "math"), "abc123",
                     "--note", "resumed on l4")
        merged = LocalStore(self.top / "opd" / "math").read_annotations()
        self.assertEqual(merged["abc123"],
                         {"name": "first", "tags": ["opd"],
                          "note": "resumed on l4"})

    def test_the_runs_view_reads_the_tag_back_and_greps_it(self) -> None:
        self.run_cli("tag", str(self.top / "opd" / "math"), "abc123",
                     "--name", "teacher 32B", "--tag", "opd")
        listed = self.run_cli("runs", str(self.top))
        self.assertIn("folder opd/math", listed)
        self.assertIn("teacher 32B", listed)
        self.assertIn("abc123", self.run_cli("runs", str(self.top),
                                             "--grep", "TEACHER"))
        self.assertNotIn("abc123", self.run_cli("runs", str(self.top),
                                                "--grep", "gspo"))


if __name__ == "__main__":
    unittest.main()
