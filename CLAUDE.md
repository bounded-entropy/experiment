# rlstack — session handover

Samarth's personal high-throughput RL-for-LLMs harness ("the thin wrapper").
Designed and built across long Claude sessions; everything you need to
continue is in the repo.

**Experiment launch rules are in `AGENTS.md` and apply to every agent and
session.** All experiments, including recovery and controls, go through the
standing desk. A failure in that path is a reason to repair it, not to create
a private runner. Check relevant implementation branches before duplicating
an accepted ADR's work.

## Read before writing code

1. **STYLE.md** — binding. Eight rules; rule 8's folder tree IS the
   architecture, enforced by `tests/test_architecture.py`. Deviations are
   review findings.
2. **ARCHITECTURE.md** — the universal vocabulary reference (#54): every
   noun and verb defined once; docstrings speak it, and a word used with a
   different meaning is a finding.
3. **agent-context/CONTEXT.md** — the decision log. Chronological, numbered;
   later entries supersede earlier ones (#28–#48 cover the current shape:
   the trajectory/wave rename, the loss zoo + stress matrix, the Arbiter,
   the multi-tenant Learner, the Host, the operational CLI + observe/, the
   flow graph + dictionary.json, the #38 loss-purity ruling, the scoring
   verb + real opsd, the observer UI + custom panels, #43: the fleet —
   hosts as atomic partitions, join/carve/acquire, the wire — #44: the
   trainer's punica, #45: TP/transport/FSDP on metal, #46: soft prompts +
   the side_attention refusal, #47: real OPD, #48: the Lowering — one
   contract per (adapter type, side) — #54: ARCHITECTURE.md + the docstring
   recode, and #55: the vocabulary rename). If code and an early entry
   disagree, the code plus the latest entry win.
4. **agent-context/rl-stack-spec.md** — the spec canon, v3 (folded through
   #43). Invariants I1–I12. Deltas after the fold-in live in CONTEXT.md
   (incl. #55's terminology: the spec text still says gpuset/kind/llm in
   places — swap at the v4 fold).
5. **knowledge/** — the ADRs: one file per major architectural change,
   written and ANSWERED before the code exists. `knowledge/TEMPLATE.md` is
   the form; the section below is the rule.

## Working norms (Samarth's, stated across sessions)

- Ultra-readable code beats clever code; spec vocabulary goes in executable
  positions. Docstrings state the rule a thing enforces.
- One named function/method per rule; typed records, no meta-dict bags, no
  duck-typing (we own both sides of every interface).
- Spec-shape changes get a new numbered entry in agent-context/CONTEXT.md —
  and fold into rl-stack-spec.md when they change an invariant.
- The fakes suite must stay green: `python3 -m unittest discover -s tests`
  (Python ≥ 3.11 — use python3.13 locally; stdlib-only, torch/vllm/modal
  lazy). Resume-equivalence (`tests/test_resume.py`) is byte-identical run
  dirs — protect it.

## Architectural design records (knowledge/)

A **major architectural change gets an ADR before it gets code.** The ADR
records the original intent verbatim, turns it into a plan a human can read,
and ends in a list of questions Samarth answers one by one. Nothing is
implemented until every question carries an answer and the status says
`Accepted`.

**Be judicious.** The bar is: a new primitive, a new daemon or verb, a change
to an invariant (I1–I12) or to a store layout, a new folder region or a
crossing of STYLE.md rule 8's import graph, anything that changes run identity
or the resume-equivalence bytes, or a decision that will be re-litigated in six
months. A bug fix, a new loss or postprocessor in an existing shape, a test, a
deploy script, or a rename does NOT get an ADR — it gets a commit. Writing an
ADR for a small change devalues the ones that matter; skipping one for a large
change costs a rewrite. If it is genuinely borderline, ask Samarth rather than
guessing.

**The form** — copy `knowledge/TEMPLATE.md` to
`knowledge/NNNN-kebab-title.md` (next free number, zero-padded to four):

- **Meta** — date, title, status (`Proposed` → `Answered` → `Accepted` →
  `Implemented`, or `Rejected` / `Superseded by ADR NNNN`), the folders it
  touches, the invariants it bears on, the CONTEXT entries it extends.
- **Original prompt** — Samarth's words, verbatim and uncut.
- **Context / problem** — what is true today, what breaks, why the existing
  primitives do not already answer it. File and line, and the measurement if
  one exists.
- **Decision** — the shape in the repo's own vocabulary (ARCHITECTURE.md),
  then: **touched vs. untouched** files (the untouched list is the blast
  radius, each with its reason); **promises vs. non-promises** (what is true
  after the commit, in checkable terms — and what it deliberately does not do
  or prove on metal); **interfaces** (which protocol, which verb, which gate
  check, which store key, what `observe/` sees); and **sketches** — the two or
  three signatures that make "agree" a decidable question, not the
  implementation.
- **Questions** — numbered, each a real fork with a recommendation and the
  consequence of the other branch. Cover the races, the crash-midway state, the
  resume path, and the byte-identity obligations explicitly; a question with an
  obvious answer is padding, and an unasked race is a rewrite. Each ends with a
  blank `> **Samarth:**` line, answered `agree` or `disagree` plus reasoning.
- **Outcome** — filled at implementation: what landed, what the answers
  changed, the test count, what stayed unproven, the CONTEXT entry number.

**The loop.** Write the ADR and commit it at `Proposed` — the commit is the
handoff, not a draft in chat. Samarth answers inline and commits; the agent
re-reads, folds every `disagree` into the Decision (a disagreement that changes
the shape may open new questions — ask them rather than assuming), and marks
`Accepted`. Only then does code get written. Do not implement against an
unanswered question and do not silently take the recommendation.

**Its relation to CONTEXT.md.** The ADR is the decision BEFORE the code; the
CONTEXT entry is the record AFTER it — what actually landed, what metal proved,
what stayed open. Both exist for a spec-shape change: the ADR is where the
questions were answered, the numbered CONTEXT entry is the canon that later
sessions read. Cross-reference each in the other, fold into
`agent-context/rl-stack-spec.md` when an invariant moved, and set the ADR to
`Implemented`. A superseded ADR is never edited away — it gets
`Superseded by ADR NNNN` and stays, because the reasoning is the artifact.

## Quick commands

```
python3.13 -m unittest discover -s tests        # fakes suite (~8s; torch-gated cases skip)
PYTHONUNBUFFERED=1 modal run deploy/steer_l4.py::run_tests   # the same suite inside the image
modal run deploy/steer_l4.py::probe             # the steer's parity exam on one L4 (~4m)
modal deploy deploy/desk.py                     # THE desk — once, for every venue (ADR 0007)
modal run deploy/desk.py::status                # every listing and every metal, one fleet
modal run deploy/desk.py::recipe --metal steer-l4 --engine ... --learner ...
                                                #   what that metal builds (a metal boots bare)
modal run deploy/desk.py::sweep                 # release every metal the desk holds (guarded)
modal deploy deploy/steer_l4.py && \
PYTHONUNBUFFERED=1 modal run deploy/steer_l4.py::check       # two tenants through the desk,
                                                #   released by the desk (~10m; ADR 0004)
modal run deploy/stress_fleet.py::topology      # three adapter types at tp=2/fsdp=2
modal run deploy/concept_steer.py::smoke        # the image, exercised, BEFORE any deploy (ADR 0008)
RLSTACK_OBSERVER=https://<workspace>--rlstack-ui.modal.run \
modal run deploy/concept_steer.py::train --layer 10 --teacher-run <rid>   # ADR 0005, UNRUN
                                                #   a campaign door follows its run at the
                                                #   observer now, so it needs the URL (ADR 0008)
python3.13 -m rlstack ui <store-root>           # the observer UI over a local store
```

The campaign deploys (gsm/dsl/sweeps) left the tree with the gsm campaign;
recover one from history (`git show 6703274^:deploy/gsm_a100.py`) when a venue
needs its shape. Never stop a `modal run` driver mid-call: the cancellation
propagates into the metal container and kills it (memory: CONTEXT #77).
