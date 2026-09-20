<!-- ADR formatting: keep each prose paragraph and list item on one source line. -->

# ADR 0020 — Verified immutable reads and a shared pod cache; runners own prefetch and publication timing

| | |
|---|---|
| **Date** | 2026-09-19 |
| **Status** | Implemented |
| **Author** | Claude and Codex, with Samarth |
| **Touches** | `data/stores/{base,strangeloop,address}.py`, Strange Loop deploy configuration and worker wiring, examples, tests |
| **Invariants** | I3 and I10 unchanged: object hashes and the authoritative store remain the same. Hashed payload publication uses size confirmation; readers verify content. |
| **CONTEXT** | Amends ADR 0015 for CAS blobs and ADR 0019 named-adapter payloads; CONTEXT #95. |

## Original prompt

> could you look through strangeloop's code and see why their scratch is so slow? i was under the impression that it was just a modal volume that was mounted, so im confused why it's so slow

> yes please write the ADR with the plan to make the scratch reads and writes faster

Additional approval, after reviewing the revised responsibilities:

> i agree with all you said above. could we implement the store fixes fully? i approve the new adr 20

## Context / problem

Strange Loop scratch is a Modal Volume mounted on each lease. Each mount has its own snapshot; commit publishes writes, but does not refresh other mounts. The existing Strange Loop store reads everything through HTTP. Mounted writes call `sync /scratch` and download the whole payload through HTTP to verify publication. Named adapters cost both a large read and a large verification download, repeated across processes despite an in-process LRU.

The native Modal store already avoids reload: its module docstring records reload disrupting concurrent writers in a previous twelve-arm run. It reads its mount first and falls back to `volume.read_file()` on a miss. Native reads working well did not establish that reload is safe.

The 2026-09-19 bounded CPU storage drill used synthetic 61,361,056-byte objects. Four concurrent HTTP reads reached 51–103 MB/s across eight-file batches; native Modal SDK reads reached 59–66 MB/s across four-file batches in a different account and environment. First mounted batches reached 85–97 MB/s before refresh overhead; warm local cache reads were much faster (including OS page cache). These measurements do not establish a universal winner for first reads. Individual commits took 1.22–1.48 s; eight-file commits took 1.47 and 3.56 s, suggesting batching can reduce application waiting without proving transaction semantics. Detailed samples and methodology are retained in `research/storage-throughput-2026-09-19/` in the parent experiment workspace.

Refresh was refused with an open scratch file, a surviving mmap, or an open `/persist` log. Successful refresh reported both `/scratch` and `/persist`. The current worker redirects its service log onto `/persist` for its lifetime. A store read cannot coordinate all other users of these mounts. There was no live training, concurrent-writer safety, or crash-atomicity proof.

## Decision

The store optimizes access to bytes whose expected sha256 is already known: CAS blobs and write-once named-adapter payloads. Run configuration and shared runners own which future inputs to prefetch and when to publish completed outputs. Pod-level coordination owns any future refresh. This ADR adds no speculative fetching, refresh calls, delayed publication, or batch interface.

### Read and write rules

```
named payload: verified in-process LRU → authoritative named meta on an LRU miss
hashed bytes: shared pod cache → already-visible mount → authoritative HTTP
```

Every returned payload is hash-checked, including in-process hits. A corrupt cache entry is discarded and a corrupt or unavailable mount falls through without modifying the mount. An authoritative hash mismatch raises `StoreError` with the key and expected/actual hashes. A missing name returns `None`; a sealed name with a missing payload is an error. Misses are not cached. A mount can serve any matching hashed object, regardless of which pod wrote it. Mutable metadata, journals, ledgers, checkpoints, listings and stats keep ADR 0015's authoritative path.

Hashed writes use the existing publication mechanism (mounted write plus sync, or HTTP upload) and confirm an authoritative file stat with the offered length, without downloading the payload. Definite rejection can retry; a lost acknowledgement never replays the mutation and is resolved only by observation. A same-length corrupt payload can pass write confirmation and will fail at a subsequent uncached authoritative read. Named metadata is still written after confirmed payload publication, fully read back, and followed by the existing persist boundary. Size confirmation is not a content or atomic-transaction guarantee.

### Cache and configuration

`HashedReads` declares a local cache directory, its byte bound, and whether already-visible mount reads are enabled. Defaults disable both optimizations; read verification and size-confirmed writes still apply. Strange Loop metal configuration passes these settings through the worker environment, inherited by spawned residents. Reopened residents remain read-only, use the same cache, and may consult the root already carried by `StoreAddress`; no provider credentials or tuning enters run identity or stored run bytes.

`BlobCache` lives in the Strange Loop backend. Content lives at `<directory>/<sha256[:2]>/<sha256>`. A process-shared file lock covers a cache miss through fetching and insertion, so simultaneous requests for retained content share a download. Bounded lock stripes avoid an unbounded lock-file index. A separate short cache-maintenance lock protects insertion, eviction and orphan temporary cleanup; it is never held across network I/O. Complete bytes are installed by temporary file and rename. Crash releases locks; partial bytes are never returned. Cleanup cannot remove another active insertion's temporary. Eviction bounds retained payload bytes and preserves already-open readers under POSIX unlink semantics. Objects larger than the bound are served without disk retention. The cache must be outside shared mounts. Cache I/O failure falls through to verified source reads, without masking authoritative errors.

### Touched / untouched

- **Touched:** base Store gains protected `_read_hashed(key, sha256)` and `_write_hashed(key, data, sha256)`; CAS and named payloads use them. Other backends gain read verification through the default implementations.
- **Touched:** Strange Loop implements stat-confirmed hashed publication, shared caching and optional mount reads; resident reopening and deployment configuration carry the settings.
- **Touched:** tests exercise corruption, the three publication outcomes, metadata sealing, independent processes, eviction, crash cleanup, configuration and byte-equivalent recovery. The opt-in live drill exercises the implemented hashed paths without refreshing mounts.
- **Untouched:** mutable publication and single-owner custody; checkpoints and their cadence; runner scheduling; the fit/census output shape; run identity and persistent layout. `tests/test_resume.py` remains unchanged.

### Promises / non-promises

- A returned CAS or named payload has the expected hash. Cache/mount hits avoid payload HTTP traffic, including across processes sharing a cache.
- Hashed publication does not download its payload for confirmation. Small named metadata keeps full verification. Public Store verbs keep their signatures and durability meanings.
- No read performs refresh. Other pods' new objects remain readable through HTTP fallback and are cached locally on demand.
- No automatic prefetch, parallel upload pool, commit batching/coalescing, checkpoint optimization or census rewrite lands here. Deliberate batching needs runner policy and a future narrow store operation preserving payload-before-seal ordering; looping over `write_named` does not batch its existing commit boundaries.
- No end-to-end GPU throughput or provider failure-atomicity claim follows from the bounded storage measurements or offline tests.

### Interfaces

```python
class Store:
    def _read_hashed(self, key: str, sha256: str) -> bytes: ...
    def _write_hashed(self, key: str, data: bytes, sha256: str) -> None: ...

@dataclass(frozen=True)
class HashedReads:
    blob_cache_dir: Path | None = None
    blob_cache_bytes: int = 0
    mount_reads: bool = False
```

Store notes identify cache, mount or API sources and rejected local copies; run files acquire no new telemetry or semantics.

## Questions and approved answers

The answers below record Samarth's approval of the revised proposal in conversation; they are not approvals of the superseded refresh-on-miss proposal.

**Q1. Confirm hashed writes by size and verify content at read?** Yes. Same-size corruption moves detection to a reader. Mutable objects and the named seal keep full readback.

**Q2. What happens on authoritative corruption?** Raise `StoreError`; do not return `None` or rewrite the authoritative object.

**Q3. All hashed objects or a size threshold?** All CAS and named payloads; no threshold.

**Q4. Shared bounded cache?** Yes, with duplicate-download coordination, atomic insertion and safe eviction; directory and capacity are deployment settings.

**Q5. Which mounted objects may be read?** Any already-visible object matching its expected hash, including other pods' objects. No ownership index is needed.

**Q6. Implement automatic refresh behind a setting?** No. Remove refresh from this ADR's read path entirely, in light of the open-file drill and worker log lifetime.

**Q7. Who owns refresh and prefetch?** Any future refresh needs pod-level coordination. Run configuration and the shared runner own optional lookahead; the store performs ordinary reads and caching.

**Q8. Include census changes?** No. Probe-only jobs and eliminating unchanged adapter copies belong to runner work outside this implementation.

**Q9. Add parallel uploads or batched commits?** No. Keep existing commit boundaries and the write lock. Runner-directed publication batches and a supporting store operation are follow-up work.

## Outcome

Implemented in `claude/dreams` after Samarth approved the revised scope. CAS and named payloads verify their hashes; Strange Loop uses stat-confirmed hashed writes, a bounded cache shared across processes, and optional already-visible mount reads. The pod settings reach spawned residents through inherited environment. Example metals enable an 8 GiB local cache and mount reads. Mutable records, named seals, commit cadence and runner scheduling retain their protocols. No refresh endpoint was added.

Validation: 1,918 local tests completed with 255 environment-gated skips and no failures; `tests/test_resume.py` and the fourteen canonical venue rows are unchanged. Five live provider storage tests plus 122 storage tests passed on a Strange Loop CPU lease (127 total, no skips). The first CPU pass found the mount-symlink validation bug; the fix and a regression case passed on both platforms. Tests establish size-only payload confirmation, full named-meta confirmation, API/mount/cache reads, corrupt-source fallbacks, cross-process download coordination, eviction, killed-process cleanup, and byte-equivalent recovery. Initial failed validation is retained alongside final per-test timestamps, outcomes and timings in [the evidence](evidence/0020-store-validation.json).

No GPU experiment, production deployment, provider failure-atomicity test or concurrent live writer drill was performed. Test payloads, staging archives and temporary credential copies were cleaned up; the CPU lease was released. CONTEXT #95 records the implementation.
