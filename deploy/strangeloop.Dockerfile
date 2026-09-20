# Strange Loop appends this fragment to its fixed runner FROM image.
# Build on the image service, before allocating a GPU. This image is unvalidated.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /usr/local/bin/uv
ENV UV_NO_CACHE=1
RUN uv python install 3.12 && uv venv --python 3.12 /opt/rlstack
RUN uv pip install --python /opt/rlstack/bin/python vllm==0.28.0 torch==2.13.0 transformers==5.16.1 safetensors numpy wandb jinja2
RUN /opt/rlstack/bin/python -c "import torch, transformers, safetensors, numpy, wandb, jinja2; print(torch.__version__, transformers.__version__)"
ENV VLLM_USE_FLASHINFER_SAMPLER=0
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV OMP_NUM_THREADS=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/scratch/cache/huggingface
ENV TORCH_HOME=/scratch/cache/torch
# Missing weights/tokenizers should fail; stage them with CPU scratch sync first.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
