# A Desk with RunPod metal

rlstack does not create or stop RunPod pods and holds no RunPod key ([ADR 0021](../knowledge/0021-runpod-metal-is-started-by-hand-and-attaches-to-a-desk-beside-its-volume.md)). You start every pod by hand; each GPU pod's worker attaches itself to the desk. You also stop every pod by hand: **an idle or released pod keeps billing until you stop it in the RunPod console.** Nothing here has run on a RunPod pod yet.

The code is grouped like the other venues: `rlstack/runner/venues/runpod/provider.py` holds the config and the refusing provider, `desk.py` the desk pod's services, `worker.py` the GPU process. They share `runtime.py` and `serving.py` with Strange Loop and Modal (ADR 0017).

## Layout

- One **network volume**, in one data center, mounted at `/workspace` on every pod. It holds the store (`/workspace/rlstack`), the desk's state dir (`/workspace/rlstack-desk`: `desk.lock`, `http-token`), the source tree (`/workspace/rlstack-src`), the weight cache (`/workspace/cache/huggingface`) and the process logs (`/workspace/rlstack-logs`).
- One **CPU pod** runs the desk and the observer. Expose its desk and observer ports (18760, 8321) as HTTP ports so the operator can reach them through RunPod's HTTPS proxy.
- One **GPU pod** per configured metal slot. Expose no ports. Start it with `RLSTACK_METAL=<slot name>` in its environment.
- **Global networking** on for every pod. Pods reach each other at `http://<podId>.runpod.internal:<port>`; those are the addresses in the config and in the desk's journal. Do not use proxy URLs for metal: the proxy closes requests after about 100 s, and a carve or a long generation is one held request.

## Steps

1. Build and push the image: `docker build -f deploy/runpod.Dockerfile -t <registry>/rlstack-runpod:<tag> .`
2. Create the network volume. From any pod that mounts it, put the repo at `/workspace/rlstack-src` (for example `git clone`) and stage the model weights under `/workspace/cache/huggingface`; the image runs offline.
3. Start the desk's CPU pod from the image with the volume and global networking, and without `RLSTACK_METAL`. Its start command runs `python -m deploy.runpod --config $RLSTACK_DESK_CONFIG serve`.
4. Fill `examples/runpod-h100.json` (or your own file, named by `RLSTACK_DESK_CONFIG`): `desk_endpoint` from the desk pod's id, `operator_endpoint` from its proxy URL, and each slot's `endpoint` from its GPU pod's id. Pod ids are known once a pod exists, so start the GPU pods and fill their endpoints before step 5; a worker whose slot or endpoint is wrong registers an address nobody can dial.
5. Start each GPU pod with `RLSTACK_METAL=<slot>`. Its worker reads the same config from the volume, waits for the desk, and registers with `$RUNPOD_POD_ID` as its container. The worker reads the desk's token from `/workspace/rlstack-desk/http-token`.
6. From the laptop, export the token once (copy it from the desk pod's `/workspace/rlstack-desk/http-token`) and use the operator verbs:

```sh
export RLSTACK_HTTP_TOKEN=...
python3.13 -m deploy.runpod --config examples/runpod-h100.json status
python3.13 -m deploy.runpod --config examples/runpod-h100.json submit spec.json --subdir pilot --every 8
python3.13 -m deploy.runpod --config examples/runpod-h100.json stop --run <id>
python3.13 -m deploy.runpod --config examples/runpod-h100.json release rp-h100-a
```

`python3.13 -m rlstack fleet --desk <operator_endpoint> ...` works too. The observer is at the observer port's proxy URL; open it once with `?token=<token>`.

## Releasing and stopping

The desk's idle clock is on, with a finite `idle_s`: an idle metal is released through the desk, its residents drain, and its worker exits. The pod does not stop; its start command sleeps so that a container restart cannot re-register a released worker. Stop the pod in the RunPod console. `release` prints the same reminder, because the desk reports `terminated: false` for every RunPod release.

To bring a slot back, restart its pod; the worker registers a new epoch. The desk never knocks a RunPod metal (no metal is bootable; a knock is journaled as `knock-refused`), and automatic recovery is off. A run on a metal that died stays parked until its pod is back and you resubmit with `--resume`. Confirm the old pod is stopped first: rlstack cannot confirm it for you.

## Unverified

Every RunPod fact this relies on is unmeasured: global networking's availability for CPU pods and its open ports, the network volume's cross-pod visibility and `flock`, `RUNPOD_POD_ID`, the container restart policy, the proxy's time cap, and whether the image's dependency layers build on the chosen CUDA base. ADR 0021 lists them.
