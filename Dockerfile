FROM runpod/worker-comfyui:5.8.5-base

ENV PYTHONUNBUFFERED=1 \
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
RUN comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors --relative-path models/diffusion_models --filename wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors --relative-path models/diffusion_models --filename wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors --relative-path models/diffusion_models --filename wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors --relative-path models/diffusion_models --filename wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors --relative-path models/text_encoders --filename umt5_xxl_fp8_e4m3fn_scaled.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors --relative-path models/vae --filename wan2.2_vae.safetensors && \
    comfy model download --url https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors --relative-path models/vae --filename wan_2.1_vae.safetensors

COPY comfy_client.py handler.py workflow_builder.py /worker/
COPY workflows/ /worker/workflows/

CMD ["python", "-u", "/worker/handler.py"]
