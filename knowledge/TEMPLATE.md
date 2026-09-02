<!-- Copy to knowledge/NNNN-kebab-title.md, fill every field, delete this line. -->

# ADR NNNN — <title: the decision, not the topic>

| | |
|---|---|
| **Date** | YYYY-MM-DD |
| **Status** | Proposed \| Answered \| Accepted \| Implemented \| Rejected \| Superseded by ADR NNNN |
| **Author** | <session/agent> |
| **Touches** | <folders under STYLE.md rule 8, e.g. `runner/daemons/`, `data/stores/`> |
| **Invariants** | <I1–I12 this bears on, or "none"> |
| **CONTEXT** | <entries this supersedes or extends; the entry number once implemented> |

## Original prompt

> Verbatim. What Samarth actually typed, uncut and un-paraphrased. If the ask
> arrived across several messages, quote each in order.

## Context / problem

What is true today, what breaks, and why the existing primitives do not already
answer it. Ground in files and line numbers. State the measurement if there is
one; say "unmeasured" if there is not.

## Decision

One paragraph: the shape of the answer, in the repo's vocabulary
(ARCHITECTURE.md). Then:

### Touched / untouched

- **Touched** — `path/file.py`: what changes and why it is the right region
  under STYLE.md rule 8.
- **Untouched** — the files a reader would expect to change and that do not,
  each with the reason. This list is load-bearing: it is the blast radius.

### Promises / non-promises

- **Promises** — what is true after the commit, in checkable terms ("the same
  bytes whether X or Y", "the suite is green", "refused at the gate").
- **Non-promises** — what this deliberately does not do, does not measure, and
  does not prove on metal. Anything landing UNPROVEN says so here and in
  CONTEXT.

### Interfaces

How it meets the primitives that already exist — which protocol it implements,
which verb it adds, what the gate checks, what the store key looks like, what
`observe/` sees. Name the seam, not the plumbing.

### Sketches

```python
# The two or three signatures that carry the design. Not the implementation —
# enough that "agree" or "disagree" is a decidable question.
```

## Questions

Numbered, each a real fork with a stated recommendation and the consequence of
each branch. Cover the races, the resume path, the crash-midway state, and the
byte-identity obligations. Samarth answers each with **agree** or **disagree**
plus reasoning; nothing is implemented until every one is answered.

**Q1. <the question>**
Recommendation: <the branch this ADR would take, and why>.
If the other branch: <what changes>.

> **Samarth:**

**Q2. <…>**
Recommendation: <…>.

> **Samarth:**

## Outcome

Filled at implementation. What actually landed, what the answers changed, test
counts, what stayed unproven, and the CONTEXT entry number that records it.
