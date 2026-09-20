# A local Desk with Strange Loop metal

This backend runs the existing Desk and observer on your laptop. The operator explicitly acquires configured GPU slots; each pod runs the shared MetalService and resident processes. Submissions still use RemoteDesk and the normal placement, custody, code-agreement and resume gates. This branch has local tests; the image build, reverse SSH route and full live Store/Desk drill have not been run.

The implementation is grouped by responsibility: `rlstack/runner/venues/strangeloop/provider.py` manages the provider and SSH routes, `desk.py` starts the laptop services, and `worker.py` starts the GPU service. Modal has matching `provider.py`, `desk.py` and `worker.py` modules. Both backends use `rlstack/runner/venues/runtime.py` for their lifecycle and `rlstack/runner/venues/client.py` for submission, readiness and progress helpers (ADR 0017). The independent metric exporter is `rlstack/runner/exporters/wandb.py`; HTTP and blob transfer live in `rlstack/runner/transports/`. The existing Desk, Host and resident interfaces retain their usual modules.

Create a configuration from `examples/strangeloop-config.json`. Replace the scratch account identity with the one reported by your login, choose the allowed metal slots, set the account-wide GPU ceiling explicitly, and select the build recipe your experiment needs. Do not run two desks against the same scratch prefix. The local state directory owns provider lease IDs, idempotency labels, a process lock and a private HTTP token; it is operational state, outside run identity.

Build a compatible image before taking a GPU. The fragment extends Strange Loop's fixed runner base and installs the repo's pinned Python/vLLM/torch stack into its own environment. It deliberately retains the provider runner and mounts. The build has not yet established CUDA, driver or compiler compatibility.

```sh
strangeloop whoami
strangeloop image list
strangeloop image build rlstack-py312 --dockerfile deploy/strangeloop.Dockerfile --wait
```

Before acquiring metal, use `strangeloop scratch` to inspect existing datasets and weights, and `strangeloop scratch sync` for missing downloads on the provider CPU service. The image uses `/scratch/cache/huggingface` and disables online Hugging Face fallback: the weight and tokenizer cache must resolve the model names in the spec before a GPU run can start. The first live drill must verify that cache layout; scratch download completion alone does not prove a compatible Hugging Face cache.

From the worktree root, keep the local service running:

```sh
python3.13 -m deploy.strangeloop --config /absolute/path/strangeloop.json serve
```

In another terminal, acquire a configured slot, inspect registration, and submit a canonical spec using its exact run subdirectory. The `up` command only returns after the worker's current provider lease and epoch are registered with the Desk. This is a capacity acquisition; the experiment itself enters through `submit`.

```sh
python3.13 -m deploy.strangeloop --config /absolute/path/strangeloop.json up sl-a100
python3.13 -m deploy.strangeloop --config /absolute/path/strangeloop.json status
python3.13 -m deploy.strangeloop --config /absolute/path/strangeloop.json submit /absolute/path/spec.json --subdir pilot --every 1 --delivery wire --objective train/loss
```

`--objective` names an actual scalar in a committed ledger row. It is required because the exporter must not invent a headline number. The submission command declares the accepted run and starts a separate offline W&B exporter on its anchor's lease. The exporter reads complete committed ledger lines; it never changes training data, the ledger or the manifest. Its ready file supplies the exact offline bundle to `gpu declare`. Missing objective values are reported, not replaced with fake values. An exporter restarted explicitly uses a new offline bundle and replays committed history; W&B offline mode does not provide reliable resume. Any other client using RemoteDesk directly must arrange the same export/declare step.

The observer is at `http://127.0.0.1:8321` by default. A run's returned `run_ref` is the exact path for follow-up reads. The GPU service log is under `$PERSIST_DIR/runs/<lease>/rlstack-service.log`; persistent scientific data is in `/scratch/rlstack`, and exported W&B bundles are in the lease's artifact directory.

## Addresses and file publication

Every recorded service address uses the laptop gateway, for example `http://127.0.0.1:18760#sl-a100/host-name@epoch`. A local SSH forward connects the gateway to each GPU service. The same SSH connection reverse-forwards port 18760 on that GPU to the laptop gateway, so another GPU can use exactly the same address. A sibling host in the same process uses the existing in-process switchboard. The gateway only relays HTTP frames; admission, model state and mutation execution remain on the serving host.

This requires OpenSSH reverse forwarding, in addition to Strange Loop's documented local forwarding. The worker refuses startup if its reverse route cannot reach the desk. The first live lease (2026-09-13) exposed `AllowTcpForwarding local` in the runner's OpenSSH configuration. The provider now validates and reloads that lease-local configuration with reverse forwarding restricted to the configured Desk loopback port (`PermitListen 127.0.0.1:<port>`); `GatewayPorts no` and authentication stay intact. The GPU-to-laptop round trip passed on A100-80GB. This setup currently depends on the runner's `/run/strangeloop-ssh/sshd_config` layout; native support for reverse forwarding would remove that dependency. The HTTP transport now offloads large requests and results to verified temporary blobs at each hop. The defaults are 2 GiB per serialized value and an 8 GiB spool budget per daemon; set `RLSTACK_HTTP_MAX_BLOB_BYTES` and `RLSTACK_HTTP_SPOOL_BYTES` before starting the local service and clients to change them. Bootstrap propagates those settings to workers. Temporary disk and Python memory are needed on the laptop as well as workers. A real 65 MiB request/result has passed through both local HTTP hops; live SSH throughput remains unmeasured. A result-download retry never repeats its RPC, while a lost RPC acknowledgement remains an ambiguous operation. See [ADR 0016](../knowledge/0016-http-rpc-transfers-use-temporary-blobs.md).

Workers write locally and call checked `sync /scratch` for each mutation. CAS and named-adapter payloads are confirmed by API file size and verified by hash when read; named metadata and mutable records still receive full readback. Before registration, each worker performs an isolated publication check. Mutable reads and stats remain API-authoritative. Hashed reads can use a shared local cache and any already-visible matching mounted bytes, then fall back to HTTP. No read refreshes the mount. Credentials enter the worker through a private transfer and its inherited token-file setting; source/config transfer is not a dataset download. See [ADR 0020](../knowledge/0020-hashed-bytes-are-verified-at-read-cached-on-the-pod-and-read-from-the-mount.md).

Each metal may declare `blob_cache_dir`, `blob_cache_bytes`, and `mount_reads`. The examples enable an 8 GiB cache at `/tmp/rlstack-blobs` and mount reads. Use a dedicated directory on local disk, outside `/scratch` and `/persist` (including their symlink targets); the cache owns its contents and evicts old payloads at the byte bound. Omit the settings to disable these optimizations, or set the directory to `null`, bytes to `0`, and mount reads to `false`. Bootstrap exports `RLSTACK_BLOB_CACHE_DIR`, `RLSTACK_BLOB_CACHE_BYTES`, and `RLSTACK_MOUNT_READS` (`0` or `1`) to the worker and spawned residents; existing processes take new settings only after restart. Prefetch scheduling and batched publication are separate runner work.

## Ownership and stopping

GPU idle release is finite and enabled. The Desk checks the shared busy/custody rules before releasing; provider lease expiry is an additional backstop if the laptop disappears. Keep the local Desk and its tunnels running while an experiment depends on them. Stopping the local process closes communication and does not prove that the GPU or its run stopped. Lease windows are not automatically extended: use the provider's explicit extension command before a running experiment reaches its deadline.

```sh
python3.13 -m deploy.strangeloop --config /absolute/path/strangeloop.json release sl-a100
strangeloop gpu list
```

The release door also handles a known lease whose setup failed before registration. If allocation timed out before returning its ID, retain the saved label and reconcile it with the provider; a new label could rent a second GPU. A repeated `up` reuses a live lease and does not repeat an uncertain daemon-start command. Confirm actual provider release before replacing an owner or resuming with `--resume`.

Automatic reaping, registration-driven reconciliation and parked-run recovery are disabled by default (`automatic_recovery: false`). The shared recovery path now waits for provider-confirmed stopped ownership, but enabling it still requires the live termination and scratch-publication drill. An SSH timeout is not process death. The usual explicit submission and stopped-run resume gates remain. An uncertain scratch mutation quarantines that Store writer; neither a new reader nor a successful GET proves that an outstanding provider request can no longer complete later.

Native scratch append, conditional writes and operation status are still needed. The temporary journal append replaces the authoritative prefix plus one canonical line under the existing single-writer owner. Distinct workers, abrupt failure around publication, owner handoff and real adapter payload sizes all require the bounded live drill described in ADR 0015 before calling this backend validated.

The first live pilot also found that Strange Loop sets `HF_HOME=/scratch/cache/hf` at runtime, overriding the image's value. A metal's optional `environment` map explicitly supplies application settings to the worker and its residents; the pilot names its pre-staged cache there. Host filenames are encoded inside the Scratch adapter (`:` becomes `@3a`, literal `@` becomes `@40`); the observer and all shared runner APIs retain the original logical names.

## The desk hosted

`deploy/strangeloop_desk.py` runs the same `serve` inside one standing Modal container (app `rlstack-strangeloop-desk`, workspace `yu-masala-workspace`), so the laptop can be off. Nothing about leases or addresses changes: the container opens the outbound SSH to each pod, the pod reaches the gateway at its own loopback through that connection's reverse route, and every recorded address stays `http://127.0.0.1:<gateway_port>#...`. The container carries what the laptop carried — `openssh-client`, the CLI (the published 0.8.1 wheel; the package is not on PyPI), the token as `SL_API_TOKEN` from the secret `strangeloop-token` (the CLI honours it on every command, so nothing logs in), the repo's two package trees under `/root`, and one volume `rlstack-strangeloop-desk-state` mounted at `/state`: `/state/desk` is the desk's state dir (lease files, `http-token`, `desk.lock`, export records; `DeskConfig.read(..., state_dir=)` puts it in the file's place) and `/state/strangeloop` is the CLI's config dir (`STRANGELOOP_CONFIG_DIR`: its machine key under `ssh/`, the `ssh_config` it rewrites per lease). The config it serves is `examples/strangeloop-hosted.json` — the general config plus `operator_endpoint` — or the repo-relative path in `RLSTACK_DESK_CONFIG` at deploy time.

One owner per journal (ADR 0015): stop the laptop desk first. A pod is reachable only with the key of the CLI that booted it, and the desk reconnects the leases whose files it finds, so seeding the volume with the laptop's key and state lets the hosted desk take over the laptop's live leases:

```sh
# the token the laptop's CLI is logged in with, read from its profile — never
# a placeholder typed by hand (the first deploy, 2026-09-17, paused on a 401
# because a literal `slk_...` had been pasted as the secret's value)
MODAL_PROFILE=yu-masala-workspace modal secret create --force strangeloop-token \
    SL_API_TOKEN="$(python3 -c 'import tomllib, pathlib; print(tomllib.load(open(pathlib.Path.home() / ".config/strangeloop/config.toml", "rb"))["profile"]["default"]["token"])')"
MODAL_PROFILE=yu-masala-workspace modal volume create rlstack-strangeloop-desk-state
MODAL_PROFILE=yu-masala-workspace modal volume put rlstack-strangeloop-desk-state ~/.config/strangeloop/ssh strangeloop/ssh
MODAL_PROFILE=yu-masala-workspace modal volume put rlstack-strangeloop-desk-state ~/.local/state/rlstack-strangeloop desk
MODAL_PROFILE=yu-masala-workspace modal deploy deploy/strangeloop_desk.py
```

`modal deploy` prints two URLs, labelled `rlstack-sl-desk-gateway` and `rlstack-sl-desk-observer`; put the gateway's in the config's `operator_endpoint` if it differs from the example's. The gateway checks its bearer as always; the observer is guarded by the same token — open it once as `https://<observer-url>/?token=<token>` and the browser keeps a cookie. A `desk.lock` a died container left on the volume does not brick the next start: the lock records its holder, and a record from another machine or a gone process is stale (`own_local_desk`).

The operator verbs stay on the laptop. They reach the hosted desk through `operator_endpoint` and carry its runtime token in `RLSTACK_HTTP_TOKEN`, copied from the volume once (a hosted desk's verbs refuse without it rather than mint a local token the desk would not accept):

```sh
export RLSTACK_HTTP_TOKEN="$(MODAL_PROFILE=yu-masala-workspace modal volume get rlstack-strangeloop-desk-state desk/http-token -)"
python3.13 -m deploy.strangeloop --config examples/strangeloop-hosted.json status
python3.13 -m deploy.strangeloop --config examples/strangeloop-hosted.json reauth     # every slot, since the state is on the volume
```

The container was not run on Modal when this landed: the wheel install, the port check against the desk's startup (`startup_timeout` 900 s, because saved leases reconnect before the servers bind), flock on the volume (the record is the fallback), and the volume's background commits of the lease files are the things to watch on the first deploy.
