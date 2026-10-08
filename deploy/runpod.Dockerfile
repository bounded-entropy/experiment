# The RunPod image: one image for the desk's CPU pod and every GPU pod (ADR 0021).
# Build and push it yourself, then start pods from it by hand with the network
# volume at /workspace and global networking on. Unvalidated: no pod has run it.
#   docker build -f deploy/runpod.Dockerfile -t <registry>/rlstack-runpod:<tag> .
#   docker push <registry>/rlstack-runpod:<tag>
# The image carries dependencies only. The source tree lives on the volume
# ($RLSTACK_SOURCE), so the desk and every worker run the same bytes.
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*
# From here to the HF settings: deploy/strangeloop.Dockerfile's layers, unchanged.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
ENV UV_NO_CACHE=1
RUN uv python install 3.12 && uv venv --python 3.12 /opt/rlstack
RUN uv pip install --python /opt/rlstack/bin/python vllm==0.28.0 torch==2.13.0 transformers==5.16.1 safetensors numpy wandb jinja2
# flash-linear-attention: the learner's Triton kernels for Qwen3.5's gated-DeltaNet layers.
# Without it transformers falls back to a Python loop over 64-token chunks, which a
# 65k-token dreamer prompt cannot afford (post 0010). Pure Triton, nothing compiles here.
RUN uv pip install --python /opt/rlstack/bin/python flash-linear-attention
RUN /opt/rlstack/bin/python -c "import torch, transformers, safetensors, numpy, wandb, jinja2, fla; print(torch.__version__, transformers.__version__, fla.__version__)"
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV OMP_NUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Weights live on the volume; stage them before starting a GPU pod.
ENV HF_HOME=/workspace/cache/huggingface
ENV TORCH_HOME=/workspace/cache/torch
# Missing weights/tokenizers should fail rather than download on a billed GPU.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# Where the start command finds the code and the config, both on the volume.
# RLSTACK_METAL set (at pod start) runs that slot's worker; unset runs the desk.
ENV RLSTACK_SOURCE=/workspace/rlstack-src
ENV RLSTACK_DESK_CONFIG=examples/runpod-h100.json
ENV RLSTACK_LOGS=/workspace/rlstack-logs
ENV PYTHONUNBUFFERED=1
# After the process ends (a release, a crash) the pod SLEEPS: a container restart
# must not bring a released worker back, and rlstack never stops the pod — a
# person does, in the RunPod console. It bills until then.
CMD ["bash", "-c", "mkdir -p \"$RLSTACK_LOGS\" && cd \"$RLSTACK_SOURCE\" && log=\"$RLSTACK_LOGS/${RLSTACK_METAL:-desk}-${RUNPOD_POD_ID:-pod}.log\" && if [ -n \"$RLSTACK_METAL\" ]; then /opt/rlstack/bin/python -m rlstack.runner.venues.runpod.worker --config \"$RLSTACK_DESK_CONFIG\" --metal \"$RLSTACK_METAL\" 2>&1 | tee -a \"$log\"; else /opt/rlstack/bin/python -m deploy.runpod --config \"$RLSTACK_DESK_CONFIG\" serve 2>&1 | tee -a \"$log\"; fi; echo 'rlstack exited; this pod idles and bills until it is stopped by hand'; exec sleep infinity"]
