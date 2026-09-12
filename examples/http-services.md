# Calling persistent services over HTTP

`http://host:port#host-name@epoch` and `https://host:port#host-name@epoch` are
additional addresses understood by `transport_for`. They implement the existing
`Transport` contract, so `RemotePool` and `RemoteLearner` can use resident services
outside Modal's class RPC. The host fragment and epoch are optional; the empty
fragment addresses a desk or metal plane. This extends ADR 0007's transport seam
without adding a learner verb, changing run identity or changing the store.

## Serve an existing runtime

On the service's owning event loop, after normal initialization:

```python
from rlstack.runner.remote import HostService
from rlstack.runner.transports.http import HttpServer

# learner_host and inference_host are initialized by your normal venue/runtime.
services = {
    "learner": HostService(learner_host),
    "inference": HostService(inference_host),
}
async with HttpServer(services.__getitem__, port=8000):
    await runtime_finished.wait()
```

Use a single server process for this state. The HTTP layer calls `Service.serve`
for admitted work and `Service.answer` through `LocalTransport`'s off-loop bridge
for facts. Models and adapter/optimizer state remain in the host; each request
opens an independent HTTP connection, so async sampling and synchronous learner
clients may share a transport without sharing event-loop-bound client sessions.

`examples/http_daemons.py --factory my_venue:services` is a process entrypoint for
an async context manager that yields this service resolver. That factory must
retain the venue's desk registration, finite GPU idle release and resident
teardown. The entrypoint only starts and stops HTTP; it does not replace the
Modal venue chassis, the desk, shared storage or the observer. Experiments still
submit through the standing desk.

## Reach it with Strange Loop

Use Python >=3.11 (3.12 matches the pinned deployment image) and the versions in
`deploy/modal_venue.py`. The Strange Loop warm runner may have a different Python
and may not include vLLM; install the experiment environment in its own venv.
Start the server detached or supervised on the GPU, then on the CPU machine:

```bash
strangeloop gpu forward "$ID" 18000:8000
# Another terminal on the same CPU machine:
curl --fail http://127.0.0.1:18000/health
```

```python
from rlstack.runner.remote import RemoteLearner, RemotePool, transport_for

learner = RemoteLearner(
    transport_for("http://127.0.0.1:18000#learner"), admitted=True)
pool = RemotePool(
    transport_for("http://127.0.0.1:18000#inference"), base="Qwen/Qwen3-0.6B")
# Existing install/forward_backward/optim_step and sample_tokens/score_tokens calls.
```

Multiple CPU processes on that machine can reuse the same endpoint. For several
CPU machines, establish each machine's own tunnel. Loopback addresses are not
fleet-wide routes: a desk or GPU host calling another host needs its own reachable
address. Closing the tunnel leaves the daemon and lease alive; a lease ending
loses in-memory state. Checkpoint through the existing store and extend or release
the lease explicitly.

## Wire and failure behavior

`POST /call` and `POST /ask` accept
`{host, verb, payload, deadline_s}`. Replies contain either `{result: ...}` or
`{error: {type, message}}`. Unknown hosts and verbs are refused by the service.
The server checks request sizes (64 MiB) and a finite deadline, capped at the
existing build deadline by default. `/health` becomes available after the server
context enters; factories must finish required initialization before yielding.

`WrongEpoch`, `Unreachable` and common application exceptions remain raised
exceptions. Unrecognized application exception types become `RemoteError` with
the original message. No redirects or retries are followed. A deadline ends the
caller's wait; synchronous GPU work already running in a thread may still finish.
An ambiguous `optim_step` response must not be retried without application-level
request deduplication. The HTTP layer adds no exactly-once guarantee or batching.

Loopback plus authenticated SSH is the default. For a non-loopback bind, the
server requires an explicit bearer token and expects TLS at ingress. Supply the
same `RLSTACK_HTTP_TOKEN` to factory-created clients and the example server (or
pass `token=` directly to `HttpTransport` / `HttpServer`). Tokens do not belong in
addresses or the fleet journal.

## Validation

```bash
python3.13 -m unittest discover -s tests
```

`tests/test_http_transport.py` uses real HTTP sockets and tests tenant state,
concurrency, the blocking bridge, deadlines, epoch refusals, authentication and
the in-process factory shortcut. No cloud credentials are needed.

`tests/http_gpu_probe.py` is an opt-in protocol fixture: it serves a real CUDA
`TorchLearner` with two LoRA tenants and optionally a real `VllmEngine`. It does
not allocate compute or submit an experiment. The server has a 15-minute bound.
Run its `client` mode on a separate CPU machine through the forwarded endpoint
with the same `--base` value. It verifies gradient/optimizer calls, tenant
isolation, state across reconnects, repeated scores and token sampling. This
proves the RPC path, not a full desk/observer experiment or fleet migration.
