#!/usr/bin/env bash
# Baked Wan2.2 / VAE / text encoder weights (see Dockerfile).
# Requires HF_TRANSFER env from image; HF_TOKEN improves rate limits if set.
set -eu

retry_download() {
  local url=$1 relative_path=$2 filename=$3
  local attempt=1 max=5
  until comfy model download --url "${url}" --relative-path "${relative_path}" --filename "${filename}"; do
    if [ "${attempt}" -ge "${max}" ]; then exit 1; fi
    echo "comfy model download failed (attempt ${attempt}/${max}), retrying in 60s..."
    attempt=$((attempt + 1))
    sleep 60
  done
}

BASE="https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files"

retry_download "${BASE}/vae/wan2.2_vae.safetensors" models/vae wan2.2_vae.safetensors
retry_download "${BASE}/vae/wan_2.1_vae.safetensors" models/vae wan_2.1_vae.safetensors
retry_download "${BASE}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" models/text_encoders umt5_xxl_fp8_e4m3fn_scaled.safetensors
retry_download "${BASE}/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" models/diffusion_models wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
retry_download "${BASE}/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" models/diffusion_models wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
retry_download "${BASE}/diffusion_models/wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors" models/diffusion_models wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors
retry_download "${BASE}/diffusion_models/wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors" models/diffusion_models wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors
