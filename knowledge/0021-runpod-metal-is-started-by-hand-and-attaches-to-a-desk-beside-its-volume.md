<!-- ADR formatting: keep each prose paragraph and list item on one source line. Do not hard-wrap prose at a fixed column width. Preserve intentional line breaks in fenced code, tables, and diagrams. Apply this convention to all future ADRs. -->

# ADR 0021 — RunPod metal is started by hand and attaches to a desk beside its network volume

| | |
|---|---|
| **Date** | 2026-10-06 |
| **Status** | Proposed |
| **Author** | Claude (agent session, RunPod venue) |
| **Touches** | `runner/venues/runpod/` (new provider folder), `runner/venues/serving.py` (new shared module lifted out of `strangeloop/`), `runner/venues/strangeloop/` (imports the lifted pieces), `deploy/runpod.py`, `deploy/runpod.Dockerfile`, `examples/runpod-h100.json`, `examples/runpod-desk.md`, tests |
| **Invariants** | I3 and I5 unchanged (no RunPod field enters a spec, run id or stored run byte); I10 one authoritative store; ADR 0003/0008's automatic GPU release is **not** provided on this venue (see Decision, "Out of scope") |
| **CONTEXT** | Extends ADR 0007 (one desk, addresses carry the venue, the portability audit), ADR 0008 (what RunPod will not do), ADR 0015/0016 (HTTP venue, HTTP RPC with blobs) and ADR 0017 (shared Desk/Metal runtimes); no CONTEXT entry yet |

## Original prompt

> The design just mirrors the existing designs for modal and strangeloop. We just need to implement a specific desk for runpod. All of the logic should be straightforward; just make the correct abstractions.

Scope change, the same day:

> i dont want to implement the logic of provisioning new GPUs or taking down metal or whatever (should just return a note saying this isnt implemented on runpod or smth) (and thereby dont need a runpod api key)

## Context / problem

rlstack has two venues on the ADR 0017 shape. Modal (`runner/venues/modal/`) boots metal by spawning a deployed class and confirms termination through the Modal control API; its store is a Modal Volume. Strange Loop (`runner/venues/strangeloop/`) leases pods through the provider CLI, reaches them over SSH with a laptop gateway that makes every recorded address `http://127.0.0.1:<gateway>#<metal>/<host>@<epoch>`, and stores runs in scratch: a mounted volume on pods plus an HTTP API for the off-pod desk (ADR 0015). Both share `DeskRuntime` and `MetalRuntime` (`runner/venues/runtime.py`) and the `AllocationProvider` contract (`runner/venues/provider.py`).

RunPod is a third place GPUs exist. The scope Samarth set is narrower than either existing venue: rlstack does not create or destroy RunPod pods and holds no RunPod credential. A person starts each pod by hand from a prepared image; rlstack's job begins when that pod's worker registers with the desk and ends when the desk releases it.

ADR 0007's portability audit found that "a POSIX-mount venue touches two lines inside `rlstack/` (`address.py:open_store`, `observe/locate.py:store_for`)". A RunPod network volume is such a mount: it is attached at pod creation at a fixed path (`/workspace` by default) and, inside one data center, several pods can attach the same volume. If the desk and every metal mount it at the same path, the store is the existing `LocalStore` and both of those lines already answer it.

Strange Loop's desk code contains pieces that are not about Strange Loop: the runtime bearer token, the one-owner desk lock with its stale-record fallback, the token-guarded observer server, the bare-origin check and the health probe (`strangeloop/provider.py:147-155,287-378,445-449`, `strangeloop/desk.py:25-109`), and the worker's "listen, run, wait for a signal or the release" block (`strangeloop/worker.py:75-96`). A RunPod desk needs every one of them. ADR 0017 forbids one provider importing another, so either they are copied or they move to the shared layer.

What RunPod does not provide that Modal does is listed by ADR 0008 (line 34): a name that routes to the latest version and boots on demand; at-least-once replay after a container dies; a container that quietly stops fetching; capacity waits inside a call. To those, for this venue: idle scale-down, confirmed container termination, and a TLS ingress without a request-time cap (RunPod's HTTP proxy is fronted by Cloudflare, which closes a request after roughly 100 s).

Nothing here has been measured on RunPod. Every platform fact above and below comes from RunPod's documentation as remembered by the author, not from a pod; each is listed under Open questions.

## Decision

A RunPod venue is a provider folder `runner/venues/runpod/` with `provider.py`, `desk.py` and `worker.py`, built on the shared runtimes. The desk runs on a hand-started RunPod CPU pod; every metal is a hand-started GPU pod. All of them mount one network volume at the same path, so the store is a `LocalStore` there and the observer reads it beside the desk. The desk and the metals reach each other directly at RunPod private-network addresses (`http://<podId>.runpod.internal:<port>`), with the existing HTTP transport and bearer token (ADR 0016); the operator reaches the desk through RunPod's HTTPS proxy. The `AllocationProvider` exists to satisfy the contract and refuses: `boot` and `terminate` raise `NotImplementedError` with a note that pods are started and stopped by hand on RunPod. No RunPod API is called and no RunPod key is held.

### How a pod attaches

- The config names each metal slot by `name` and `endpoint`: the http origin at which the desk and every other pod reach that metal's worker. The metal's plane address is the endpoint itself; a carved host's address is `<endpoint>#<host>@<epoch>`. Nothing relays: every caller dials the same private name, which is the property Strange Loop's laptop gateway exists to manufacture.
- The operator starts a GPU pod from `deploy/runpod.Dockerfile`'s image with the network volume, global networking on, and the env `RLSTACK_METAL=<name>`. The image's start command runs `python -m rlstack.runner.venues.runpod.worker --config <file> --metal <name>` from the source tree on the volume, then sleeps.
- The worker reads the same desk config, takes its slot, waits for the desk's `/health`, opens the store, and registers through `MetalRuntime.measured` with `container=$RUNPOD_POD_ID`. Its bearer token is `RLSTACK_HTTP_TOKEN` if the pod's env has it, otherwise the desk's `http-token` file in the desk's state dir on the volume; a worker with neither refuses to start.
- The card is measured, never declared (ADR 0001 Q6): the config does not name a GPU type. "H100 x1" is what the operator chooses in the RunPod console.

### Where the store lives

On a RunPod network volume, as a `LocalStore` at an absolute path that is identical on every pod (`/workspace/rlstack` in the example). The desk is on the volume too, so the desk, observer, metals and spawned residents all read and write one POSIX tree; `open_store` reopens it in a resident through the existing `local` backend, and `observe/locate.store_for` reads it through the plain-path branch. No new store backend is written.

The alternative is RunPod's S3-compatible API over the same volume, which would let a desk run off-pod (on the laptop or hosted on Modal like the Strange Loop desk) and is ADR 0015's mounted-plus-API pattern. It needs an S3 `Store` backend (signing, listing, and an emulated append for the ledger and journals, which ADR 0015 needed a whole decision for), a second credential, and a publication drill. ADR 0007 and 0008 both list "an S3 store" as its own later ADR. ADR 0015's dual access exists because Strange Loop's desk could not mount scratch; here the desk can mount the volume, so the pattern does not fit and is not reused.

### How the desk reaches metal

Direct HTTP over RunPod's global (private) networking: each pod has a DNS name `<podId>.runpod.internal`, reachable from other pods with global networking enabled, without a public port. The wire is the existing `HttpTransport`/`HttpServer` pair (ADR 0016), unchanged: verified temporary blobs for large frames, deadlines, epoch stamping and the bearer token. Workers listen on every interface, which `HttpServer` already permits only with a token.

The two rejected doors:

- **RunPod's HTTPS proxy** (`https://<podId>-<port>.proxy.runpod.net`) gives TLS but closes a request after about 100 s. An HTTP RPC is one held `POST /call`; a carve that builds a vLLM engine, or one long generation, outlives 100 s. Changing the transport to submit-then-poll is a transport ADR, not a venue.
- **A public TCP port** (`http://<publicIp>:<mappedPort>`) has no time cap but carries the bearer token in clear over the internet, which is what `HttpServer`'s "TLS at ingress" condition exists to prevent.

The proxy is used only for the operator's door and the observer (`operator_endpoint` and the observer port), which are TLS and short. A long operator verb (a draining `stop`) may lose its HTTP reply at the proxy; the desk keeps working, and the operator reads `status` or `dispositions` again, the same as any lost reply under ADR 0016.

### Where the desk runs

On one hand-started RunPod CPU pod in the volume's data center, with the volume mounted and global networking on: `python -m deploy.runpod --config <file> serve`, which is the image's start command when `RLSTACK_METAL` is unset. This is the simplest place that can mount the store. The desk's state dir (lock, runtime token) is on the volume, so a replacement desk pod finds them, and `own_local_desk`'s holder record fences a second desk where the volume refuses `flock`. Running on the laptop or on Modal needs the S3 store above.

### Out of scope: provisioning and automatic release

`RunPodProvider.boot` and `RunPodProvider.terminate` raise `NotImplementedError("... not implemented on RunPod: start/stop the pod by hand ...")`. The desk is built with `bootable_metals=frozenset()`, so the shared Desk refuses every knock before reaching `boot` and journals `knock-refused` with its reason; `terminate` is reached from `Desk.release`, whose existing best-effort path prints the note and records `terminated: false`. Automatic recovery is off: the shared recovery waits for provider-confirmed termination, which this venue never gives.

The shared idle clock stays on with a finite `idle_s`: an idle metal is released through the Desk as on every venue, its residents drain and its worker exits. The pod does not stop. It keeps billing until a person stops it in the RunPod console. This directly gives up AGENTS.md's rule that automatic GPU idle shutdown stays enabled, and ADR 0003/0008's release backstop; it is the cost of the scope Samarth set, and it is the main operational hazard of this venue. `deploy/runpod.py release` prints the reminder when the desk reports `terminated: false`. The worker's start command sleeps after the worker exits, so a released worker does not restart and re-register under a RunPod container restart; bringing a slot back is a pod restart by hand.

### What RunPod will not do that Modal does (ADR 0008), and how each is covered

| Modal gives | RunPod | Covered by |
|---|---|---|
| A name that boots on demand | Nothing boots; pods are started by hand | Out of scope by decision. The knock is refused and journaled (`knock-refused`); a released or dead metal stays down until a person restarts its pod, and parked runs wait visibly. |
| Idle scale-down (`scaledown_window`) | A pod bills until stopped | Desk idle release drains the metal; a person stops the pod. Not covered automatically. |
| Confirmed container termination | No API call is made | `terminate` refuses; ownership stays unresolved; automatic recovery off. A person confirms the pod stopped before resubmitting with `resume`. |
| At-least-once replay after a container dies | No replay | Already owned by the Desk: journal-first submit and idempotent verbs (F4). |
| A container that quietly stops fetching | A worker process can die inside a live pod | Already owned by the Desk: leases and epochs (F1/F2) delist it within one lease. |
| Capacity waits inside a call | No call waits | The parked state (F6). |
| Encrypted RPC between containers | Plain HTTP on the private network | Bearer token; no public port on metal pods. TLS only at the operator's proxy. |
| A store mounted everywhere | Network volume, one data center | All pods and the desk in the volume's data center. |

### Touched / untouched

- **Touched** — `runner/venues/serving.py` (new, shared): the provider-neutral HTTP desk pieces lifted out of `strangeloop/`: `runtime_token`, `own_local_desk` with `LockHolder`/`lock_holder`/`boot_id`, `observer` with `observer_guard`/`ObserverServer`, `check_origin`, `health`, `LOOPBACK`/`EVERY_INTERFACE`, and `serve_metal` (listen, run the `MetalRuntime`, end on a signal or the release). STYLE rule 8's venues region already holds "provider-neutral lifecycle"; this module imports no provider.
- **Touched** — `runner/venues/strangeloop/{provider,desk,worker}.py`: import those names instead of defining them; behaviour unchanged. The strange-loop-specific pieces (gateway, relay, SSH, CLI, `operator_token`'s Modal hint, the health wait's reverse-SSH message) stay.
- **Touched** — `runner/venues/runpod/{provider,desk,worker}.py` (new): typed `DeskConfig`/`MetalConfig`, the refusing `RunPodProvider`, the desk's `serve`, the worker's `WorkerConfig` and `run`.
- **Touched** — `deploy/runpod.py` (operator CLI: serve, status, release, stop, dispositions, submit), `deploy/runpod.Dockerfile` (the strangeloop image's layers including flash-linear-attention, on a CUDA base, plus the start command), `examples/runpod-h100.json`, `examples/runpod-desk.md`; docstring lines in `data/stores/address.py` and `observe/locate.py` naming the network volume as a POSIX mount.
- **Untouched** — `data/stores/` code (no backend: the volume is a `LocalStore`); `runner/transports/` (the HTTP wire is used as is); `runner/remote.py` (`http://` addresses already parse and route); `runner/desk.py` (the knock refusal and best-effort terminate already exist); `runner/venues/runtime.py`, `provider.py`, `client.py`; the Modal venue; the `fleet` CLI in `rlstack/__main__.py`, which already reaches any desk by address; every spec, run identity and stored byte.

### Promises / non-promises

- **Promises** — `RunPodProvider.boot` and `terminate` raise `NotImplementedError` naming RunPod and the by-hand step, and make no network call. A knock on a RunPod metal is refused and journaled. A worker registers its configured endpoint as its plane address and `RUNPOD_POD_ID` as its container, and its hosts are `<endpoint>#<host>@<epoch>`. The desk and the worker open the same `LocalStore` path. The idle limit is finite and positive or the config is refused. The Strange Loop tests pass unchanged except for one patch target that moved with `own_local_desk`. No module imports another provider (`tests/test_architecture.py`).
- **Non-promises** — no pod is created, stopped or watched by rlstack; an idle or released pod bills until a person stops it. No RunPod platform fact here is validated on a pod: global networking, the volume's multi-pod consistency and `flock`, `RUNPOD_POD_ID`, container restart behaviour and the proxy's time cap are all unmeasured. No S3 store. No TLS between pods. No live drill.

### Interfaces

The `AllocationProvider` protocol is implemented as a refusal. The desk serves `DeskRuntime.service_for` directly over `HttpServer` (no gateway, no extra verbs: `boot`, `reauth` and `export` are Strange Loop's). The observer is the shared guarded observer over the `LocalStore`. Store keys, the journal, the observer's view and every desk verb are those of the other venues.

### Sketches

```python
NOT_ON_RUNPOD = ("{verb} is not implemented on RunPod: rlstack does not provision or stop pods. "
                 "{hand}")

class RunPodProvider:
    def boot(self, name: str) -> None:
        raise NotImplementedError(NOT_ON_RUNPOD.format(verb="booting metal", hand=...))

    async def terminate(self, allocation_id: str) -> bool:
        raise NotImplementedError(NOT_ON_RUNPOD.format(verb="stopping a pod", hand=...))

@dataclass(frozen=True)
class MetalConfig:
    name: str
    endpoint: str            # http://<podId>.runpod.internal:8000
    idle_s: float = 300.0
    builds: Builds | None = None
    environment: dict[str, str] | None = None

@dataclass(frozen=True)
class DeskConfig:
    store: str               # /workspace/rlstack, the same on every pod
    desk_endpoint: str       # http://<deskPodId>.runpod.internal:18760
    metals: tuple[MetalConfig, ...]
    state_dir: Path          # on the volume: desk.lock, http-token
    desk_port: int = 18760
    observer_port: int = 8321
    worker_port: int = 8000
    idle_s: float = 300.0
    operator_endpoint: str = ""   # https://<deskPodId>-18760.proxy.runpod.net
```

## Questions

**Q1. The store is a `LocalStore` on a network volume that the desk also mounts, not an S3 store read off-pod.**
Recommendation: yes. It is ADR 0007's "two lines" path, needs no backend, and keeps one authoritative store. The price is that the desk lives on a RunPod CPU pod in the volume's data center rather than on the laptop or Modal, and that `LocalStore`'s fsync-and-rename discipline is trusted on a network filesystem shared by several pods.
If the other branch (S3 API): the desk can run anywhere, at the cost of a new store backend with an emulated append, a second credential, and its own ADR and drill.

> **Samarth:**

**Q2. Desk and metal talk over RunPod global networking (`<podId>.runpod.internal`), plain HTTP with the bearer token.**
Recommendation: yes. It is the only RunPod door that has neither the proxy's ~100 s request cap (a carve or a long generation is one held POST) nor a token in clear on the internet.
If the proxy instead: every RPC over ~100 s fails, which includes every carve that builds an engine; making that work is a submit-and-poll transport change. If public TCP instead: works today in every data center with public IPs, but the token that admits every verb crosses the internet unencrypted.

> **Samarth:**

**Q3. No gateway relay: a metal's recorded address is its own private endpoint.**
Recommendation: yes. Strange Loop's gateway exists because pods could only reach the laptop through each pod's own reverse tunnel; on a private network every caller can dial every pod, so the address already means the same thing everywhere. A desk restart does not touch metal-to-metal calls.
If the other branch (route everything through the desk): the desk becomes a hop on every pool call and its restart stalls generation across the fleet, for no reachability gain.

> **Samarth:**

**Q4. The desk runs on a hand-started CPU pod, from the same image, with its state dir on the volume.**
Recommendation: yes: the simplest place that can mount the store. One image for both roles avoids a second build; the desk ignores the GPU stack. A CPU pod bills continuously, like the standing Modal desk.
If a slim desk image instead: smaller and faster to start; one more image to keep in step with the source tree.

> **Samarth:**

**Q5. Provisioning and termination raise `NotImplementedError`; knocks are refused through `bootable_metals=frozenset()`; automatic recovery is off.**
Recommendation: yes, as scoped. Raising rather than returning `False` puts the note in the desk log where a person reads it; the empty bootable set makes the refusal a journaled `knock-refused` row instead of a swallowed exception.
If `boot` returned quietly instead: the Desk would believe a knock succeeded and wait for a registration that never comes.

> **Samarth:**

**Q6. Idle release drains the worker, the worker does not restart, and a person stops the pod.**
Recommendation: yes, with the hazard stated in the guide and printed by `release`. The desk-side idle limit stays finite so residents and their GPU memory are freed and the desk stops placing on the metal.
If the worker stayed registered instead: an idle pod would keep its residents loaded indefinitely, and nothing would tell the operator it is idle. If rlstack stopped the pod itself, it would need the RunPod key this scope excludes.

> **Samarth:**

**Q7. The source tree lives on the volume and every pod runs it from there; the image carries only dependencies.**
Recommendation: yes. Desk and workers then run the same bytes, a code change needs no image rebuild, and nothing is shipped per boot (Strange Loop's `source_bundle` exists because its pods have no shared tree). The operator updates the tree on the volume (for example `git pull` from the desk pod) and restarts the pods that should run it.
If baked into the image instead: each code change is an image build and push, and a desk and a worker started from different tags disagree silently.

> **Samarth:**

**Q8. Lift the HTTP desk plumbing out of `strangeloop/` into `runner/venues/serving.py`.**
Recommendation: yes; it is the ADR 0017 rule applied to a third provider. One test patch target moves (`fcntl` is now imported by `serving`). STYLE.md rule 8's venues line should name `runpod/` and `serving.py`; that line is Samarth's edit.
If copied instead: two copies of the lock, the token and the observer guard, which drift.

> **Samarth:**

### Open questions (not decidable without a RunPod pod)

1. Global networking: which data centers and pod types support it, whether CPU pods do, and whether ports between pods are open without being listed as exposed ports.
2. Network volume semantics across several pods: visibility latency of a write on another pod, whether `fsync` plus `rename` is atomic to other readers, and whether `flock` works (the desk lock falls back to its holder record if not).
3. `RUNPOD_POD_ID` is assumed to be set inside every pod. If it is not, the worker refuses to start; set it by hand in the pod's env.
4. Container restart policy: whether RunPod restarts a container whose start command exits. The start command sleeps after the worker to be safe either way.
5. The proxy's request cap (assumed ~100 s) and whether a request is cancelled server-side or only the client connection is closed.
6. Whether `HF_HOME` on the volume reads weights fast enough; Strange Loop's cache layout question (ADR 0020) recurs, unmeasured.
7. Whether the dependency layers of `deploy/strangeloop.Dockerfile` build on the chosen CUDA base image.

## Outcome

Implemented alongside this ADR at the user's direction, before the questions above were answered; the status stays Proposed until they are, and any `disagree` reopens the code. The implementation takes every recommendation. `runner/venues/serving.py` holds the lifted pieces and `strangeloop/` imports them (its tests changed only their import lines and one `fcntl` patch target). `runner/venues/runpod/` has `provider.py` (configs, `desk_token`, the refusing `RunPodProvider`), `desk.py` (`serve`, `desk`) and `worker.py` (`WorkerConfig.from_desk`, `run`). `deploy/runpod.py`, `deploy/runpod.Dockerfile`, `examples/runpod-h100.json` and `examples/runpod-desk.md` are the deployment side.

Validation: `python3.13 -m unittest discover -s tests` ran 1,971 tests with 257 environment skips and one failure, the pre-existing `test_strangeloop_hosted.HostedDeployTests.test_the_container_serves_the_config_from_the_volume` (it fails identically on the base commit, 1,957 tests). New: `tests/test_runpod_venue.py` (13 tests: provider refusals, config validation, token lookup, knock refusal journaled, release and idle release draining without termination, a real desk and worker over HTTP sharing one store tree with the guarded observer, the operator's release note, the image reusing the Strange Loop layers including flash-linear-attention) and one RunPod case in `tests/test_provider_parity.py`. No RunPod pod, image build, network volume or API was used; every open question above remains open. STYLE.md rule 8 still names only `modal/` and `strangeloop/`. No CONTEXT entry is written while the ADR is Proposed.
