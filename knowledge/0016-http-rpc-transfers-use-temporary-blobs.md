# ADR 0016 — HTTP RPC transfers large frames through temporary verified blobs

<!-- Numbered 0012 in the codex/strangeloop-backend worktree; renumbered on publication (ADR 0014, Part E). -->

| | |
|---|---|
| **Date** | 2026-09-13 |
| **Status** | Implemented locally; live SSH/GPU validation pending |
| **Author** | Codex with Samarth |
| **Touches** | `runner/transports/`, Strange Loop HTTP relay/bootstrap wiring, transport tests and operating guides |
| **Invariants** | I1 import boundaries, I3 unchanged identity, I5 placement-neutral bytes, I8 tenant state, existing deadline and epoch gates |
| **CONTEXT** | Extends ADRs 0007, 0008 and 0015; #91 records the publication |

## Original prompt

> ok i see nice. i agree that we should essentially make this http client ~ as robust using rpc or blobs or whatever as modal originally was. could you spawn an async subagent to deal with that? beyond that, what else needs to be done?

Samarth explicitly authorized implementation of transparent bulk HTTP transfers after the comparison with Modal's automatic blob offload. The parent delegated request and result support, bounded resources, failure tests and the real local gateway route. This authorizes the transport implementation choices recorded below; it is not a claim that Samarth separately selected each numerical default or that live GPU validation has occurred.

## Context / problem

The original HTTP PR accepts at most 64 MiB of serialized JSON in a request and reads at most that much in a response. Base64 expands binary adapter and optimizer bytes, leaving less than 48 MiB of raw binary data after JSON overhead. This restricts operations such as `load`, `emit` and `add_bundle`, even though the Engine/Learner contracts themselves impose no such limit. LocalTransport and PipeTransport have no explicit rlstack frame limit. Modal's native RPC handles large serialized arguments and results through its blob infrastructure; its documented 2 MiB inline-argument threshold is an offload threshold, not a maximum argument size.

Changing an HTTP client library or enabling HTTP chunked transfer would not change the server's total JSON limit. Splitting a learner operation into several independently dispatched operations would change its semantics. The existing Store contract returns complete byte values and the scratch API has unresolved publication/operation-status guarantees. Transport scratch must not become an experiment checkpoint or introduce a new authoritative Store.

## Decision

Keep the existing two `Transport` doors and JSON-safe model frames. Encode payloads once onto temporary files, offload large payloads to the target HTTP daemon, and invoke the service with a small reference. Serialize large results into the same daemon's temporary blob pool and return a reference. Each gateway hop performs the same protocol independently, so no caller needs another machine's filesystem path or a scratch credential. Transfer files are private operational state in the local temporary directory, never scientific Store keys or run identity inputs.

### Interfaces and ordering

1. `POST /blobs` reserves an opaque random id for a declared byte length and SHA-256. Reservations count against aggregate spool space and blob count before upload. `PUT /blobs/<id>` streams bounded pieces into that reservation. The file becomes ready only after exact length and digest validation. A failed, truncated or disconnected upload cannot dispatch an operation.
2. `POST /call` and `POST /ask` retain `host`, `verb` and `deadline_s`. Exactly one of `payload` or `payload_blob` supplies the existing JSON object. A request blob is atomically claimed once, checked again, parsed and deleted after reading. Service dispatch still uses LocalTransport and the normal epoch/admission checks. Reusing that reference is refused; this prevents reusing one uploaded frame and is not a general operation-deduplication guarantee.
3. Replies retain the existing result/error envelope. A large envelope is returned as `result_blob`. Authenticated `GET /blobs/<id>` reads only an immutable result. The client checks the declared response length and SHA-256 before returning it to the model caller. The result GET may restart once after a connection failure; the execution RPC is never repeated. `DELETE` releases transfer state after use; finite expiry collects abandoned reservations and results.
4. Every blob route uses the same bearer authorization as the RPC. References carry only validated opaque ids, lengths and digests, never arbitrary paths or URLs. Redirects and environment HTTP proxies remain disabled. The server refuses transfer encoding and conflicting/unbounded lengths rather than interpreting ambiguous framing.
5. Serialization, upload, execution and download share one caller deadline. Cancellation marks the client transfer abandoned so an upload thread cannot later invoke the operation. The local gateway propagates its remaining serving deadline through its async and synchronous relay paths; it no longer replaces that deadline with the build timeout. This metadata lives in a context variable, outside the scientific payload.

### Bounds and operator configuration

The default inline threshold is 1 MiB. One serialized payload or result may be at most 2 GiB, each daemon may reserve/spool at most 8 GiB, and there may be at most 128 blobs and 64 network handlers. Concurrent clients in one process also share an 8 GiB temporary-file budget, so constructing another HttpTransport does not bypass client disk accounting. Idle network I/O has a 30-second bound; temporary blobs expire after five minutes plus at most one collector interval when idle. Active reads/uploads finish or fail before deletion. A large model response that exceeds the byte or disk budget is reported explicitly as a transfer failure after execution, never replayed.

`HttpLimits` provides explicit constructor overrides. `RLSTACK_HTTP_MAX_BLOB_BYTES` and `RLSTACK_HTTP_SPOOL_BYTES` set the byte budgets for ordinary factory-created clients and servers. Strange Loop bootstrap copies those two values into worker configuration so the laptop, relays, GPU services and child clients share them. Raising a limit requires enough temporary disk and memory on every participating machine. `TMPDIR` selects the standard temporary-file location; do not point it at an authoritative checkpoint directory or assume temporary transfers survive daemon/machine restart.

The original 64 MiB ceiling remains a control-frame bound and an old-client compatibility bound. New peers advertise `blobs-v1`. Small calls still work with the original HTTP server; a large call to an old server fails before RPC execution because no blob endpoint exists. Old clients continue receiving inline results up to the old bound from a new server. Deploy matching code across the gateway and workers before relying on bulk transfers.

### Touched / untouched

- **Touched** — `runner/transports/http.py`: wire framing, verified transfers, finite connection admission, deadlines and error behavior; `http_blobs.py`: typed limits/references and the private spool lifecycle.
- **Touched** — `runner/venues/strangeloop/provider.py`: remaining-deadline propagation in the relay and byte-budget environment propagation during bootstrap; `worker.py`: install those operational limits before creating child processes.
- **Touched** — real-socket HTTP/gateway tests, the obsolete old-frame-limit assertion, and transport/Strange Loop guides.
- **Untouched** — Engine/Learner protocols, model codecs, LocalTransport/PipeTransport semantics, Store layout and serialization, experiment identity, optimizer semantics and tenant admission rules. The transport still materializes the application dict on each side.

### Promises / non-promises

- **Promises** — a payload larger than the original cap can cross both gateway hops and return with the same scientific bytes; corrupt or incomplete uploaded data cannot execute; a retried result download does not repeat the operation; reservations, files and network-handler counts have explicit finite limits; partial artifacts are cleaned on failure, expiry or ordinary shutdown.
- **Non-promises** — exactly-once RPC, durable operation ids/status across reconnects or restarts, resumable uploads, range downloads, automatic replay, zero-copy model codecs, bounded total Python heap, durable temporary files, cross-tenant authorization beyond the existing shared daemon token, or live GPU/SSH performance. JSON decoding, base64 and LocalTransport still materialize/copy complete values, and the stdlib JSON encoder may allocate a whole encoded string field. Spooling bounds transfer I/O and disk, not arbitrary memory allocated by a model method.
- **Non-promises** — full Modal parity. If the small RPC acknowledgement is lost before the result reference reaches the client, completion is still unknown even if a result blob exists. The caller must use existing checkpoint/ownership recovery; it cannot infer that an optimizer step did not happen. A daemon crash can leave its private temporary directory for operating-system cleanup; restart does not rediscover or replay it.

## Resolved implementation choices

**Blob location.** Use temporary storage beside the service instead of scratch: this keeps the transport provider-neutral, avoids introducing scratch publication assumptions into RPC, and grants callers no extra storage credentials. The cost is that a gateway decodes and re-encodes large values and temporary results disappear with the daemon. A future direct immutable-object-store transfer can replace this transport detail without changing learner APIs.

**Retries and interrupted uploads.** The client does not automatically retry/resume PUT or POST. A lost upload acknowledgement only leaks a bounded reservation until cleanup and cannot itself execute the model. A result GET is safe to restart against the same immutable id. Durable operation status and offset-based upload resume need their own lease/retention/idempotency design; adding an RPC retry without that design would be incorrect.

**Byte identity and crashes.** SHA-256 and explicit lengths protect transferred serialized bytes; canonical scientific bytes remain inside the existing codecs. No transfer path writes a run ledger or changes a checkpoint. The RPC remains one logical invocation after complete request validation. A timeout after dispatch may still leave synchronous model work running, so stopped-owner recovery remains a separate shared-lifecycle responsibility.

## Outcome

The implementation is present only in the isolated Strange Loop worktree. Local tests include a real 65 MiB request and 65 MiB result through the laptop gateway and worker server, plus corruption, truncation, authentication, one-use requests, concurrency, result-download interruption, lost RPC acknowledgement, deadline propagation and expiry. The HTTP-focused suite passed 38 tests. The full suite passed 1,530 tests with 194 dependency/live-environment skips and no failures; the log is `/tmp/rlstack-http-bulk-final-suite.log`. These counts include concurrent shared-recovery work in the worktree and are not a count of tests added by this transport change. No image was built, GPU allocated, deployment performed or W&B record written for this work.
