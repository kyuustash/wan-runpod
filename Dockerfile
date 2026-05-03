# syntax=docker/dockerfile:1.4
FROM runpod/worker-comfyui:5.8.5-base

ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN} \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=900 \
    PYTHONUNBUFFERED=1 \
    COMFYUI_DIR=/comfyui \
    COMFYUI_HOST=127.0.0.1 \
    COMFYUI_PORT=8188 \
    COMFYUI_INPUT_DIR=/comfyui/input \
    COMFYUI_OUTPUT_DIR=/comfyui/output \
    COMFYUI_LORA_DIR=/comfyui/models/loras \
    COMFYUI_TIMEOUT_SECONDS=1800

WORKDIR /worker

COPY requirements.txt /worker/requirements.txt
RUN pip install --no-cache-dir -r /worker/requirements.txt

# Video output and VACE helper nodes. ComfyUI includes the Wan FLF2V conditioning node in recent builds.
RUN comfy-node-install comfyui-videohelpersuite comfyui-kjnodes && \
    git clone --depth 1 https://github.com/ethanfel/Comfyui-VACE-Tools.git /comfyui/custom_nodes/Comfyui-VACE-Tools

# Wan2.2 I2V, VACE, text encoder, and VAE weights baked into the image.
# The fp8 scaled variants keep the image and VRAM requirements lower than bf16/fp16.
# Single RUN + script: one models layer (fewer large blobs during BuildKit export/push on hosted builders).
# Downloads run smaller-artifacts-first inside docker/model-downloads.sh. HF hub cache mount speeds
# rebuilds when the builder persists it; per-file retries are in the script.
COPY docker/model-downloads.sh /worker/docker/model-downloads.sh
RUN --mount=type=cache,target=/root/.cache/huggingface \
    chmod +x /worker/docker/model-downloads.sh && /worker/docker/model-downloads.sh

COPY comfy_client.py handler.py workflow_builder.py /worker/
COPY workflows/ /worker/workflows/

CMD ["python", "-u", "/worker/handler.py"]
