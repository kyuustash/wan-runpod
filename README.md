# RunPod Wan2.2 ComfyUI Worker

RunPod serverless worker for Wan2.2 image-to-video generation with ComfyUI. It exposes a compact API for:

- `flf2v`: first frame + last frame to video with `WanFirstLastFrameToVideo`
- `vace_extend`: VACE-guided continuation from the previous segment's last frame
- `workflow`: raw ComfyUI API workflow passthrough for advanced callers

The Docker image bakes in ComfyUI, VideoHelperSuite, VACE helper nodes, and the Wan2.2 fp8 model files so the endpoint does not depend on a RunPod network volume.

## What Gets Built

- `handler.py` starts ComfyUI inside the worker, accepts RunPod jobs, writes base64 inputs into ComfyUI's input folder, queues a workflow, and returns base64 output files.
- `workflow_builder.py` fills the workflow templates from request parameters.
- `comfy_client.py` talks to ComfyUI's `/prompt` and `/history` APIs.
- `workflows/wan22_flf2v.json` produces a first/last-frame segment.
- `workflows/wan22_vace_extend.json` extends the stream from a previous tail frame.

## Build And Push

Build for RunPod's Linux AMD64 runtime. **Pass a Hugging Face read token** whenever possible — it raises rate limits, cuts stalls, and avoids losing a short build budget to retry sleeps (see [RunPod builder timeouts](#runpod-builder-timeouts)).

```bash
docker build \
  --build-arg HF_TOKEN=hf_... \
  --platform linux/amd64 \
  -t <registry>/<image>:wan22 .
docker push <registry>/<image>:wan22
```

For a local build without HF auth (not recommended for cold model pulls):

```bash
docker build --platform linux/amd64 -t <registry>/<image>:wan22 .
docker push <registry>/<image>:wan22
```

The image sets `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_HUB_DOWNLOAD_TIMEOUT=900` during model `RUN` steps; `hf_transfer` is installed from [`requirements.txt`](requirements.txt). Model layers use a BuildKit cache mount at `/root/.cache/huggingface`; remote builders (including RunPod) may or may not reuse that cache across builds.

The baked model set is large. Set a large enough container disk in the RunPod template, and use a high-VRAM GPU class suitable for Wan2.2 14B fp8 video workflows.

### RunPod builder timeouts

**How to read the logs.** If you see a line like `#23 writing layer … DONE` immediately before `Build exceeded maximum time limit` (for example **1800 seconds**), that BuildKit step **succeeded** — the job hit a **wall-clock cap** on the whole pipeline (pull base, `RUN` steps, export, registry **Uploading**), not a failure inside step 23. A message such as `git was not found` comes from Buildx metadata and does not cause timeouts. **`COMFYUI_TIMEOUT_SECONDS` in the Dockerfile is only used when ComfyUI runs inside the worker**; it is not the hosted Docker build limit.

**Diagnose in the Builds UI.** On the endpoint **Builds** tab, note whether the failed run was still in **Building** (downloads and `RUN` steps) or **Uploading** (pushing multi-GB layers) when the timer stopped. **Building** → focus on Hugging Face throughput, auth, and network; **Uploading** → focus on registry speed and build timeouts that include push time.

**Shorter limits (for example 30 minutes).** Some hosted build paths enforce a **30-minute (1800s)** total budget. GitHub-connected flows can allow much longer totals; see [GitHub integration limitations](https://docs.runpod.io/serverless/workers/github-integration). If your dashboard or team settings expose a higher **build timeout**, raise it for this image. If not, treat the cap as fixed and offload the build (next bullet).

**Prefer local build + registry tag for routine deploys.** RunPod’s remote builder may not persist BuildKit cache, so redeploys can redo large downloads and pushes. Reliable approach:

1. Run `docker build` and `docker push` on your own machine, a VPS, or CI you control (use `--platform linux/amd64` and **`--build-arg HF_TOKEN`**).
2. In the serverless template, set **Container image** to that registry tag (for example `<registry>/<image>:wan22`). Redeploys that only change handler code can use cache on **your** builder instead of repeating the full cold path on RunPod.

Avoid baking multi-gigabyte model downloads in **GitHub Actions** if your account is sensitive to large outbound pulls on shared runners.

The Dockerfile uses **one `RUN` per model file** so each finished download becomes its own layer. Rebuilds can reuse those layers when the builder’s cache still has them.

## RunPod Endpoint

Create a serverless template with:

- Container image: `<registry>/<image>:wan22`
- Template type: serverless
- GPUs per worker: `1`
- Active workers: `0`
- Max workers: set to your budget
- Flash Boot: enabled if available
- Container disk: large enough for the baked image and output workspace

Then create a serverless endpoint from that template. Use `/run` for async jobs and `/runsync` only for short smoke tests.

## Request Schema

Common fields:

- `mode`: `flf2v`, `vace_extend`, or omit when passing `workflow`
- `prompt`: positive prompt
- `negative_prompt`: negative prompt
- `first_frame`: base64 PNG/JPEG data URI, used as the first frame or previous tail frame
- `last_frame`: base64 PNG/JPEG data URI, required for `flf2v`
- `width`, `height`: defaults to `832x480`
- `frames`: defaults to `81`
- `fps`: defaults to `16`
- `steps`: defaults to `20`
- `cfg`: defaults to `5.0`
- `shift`: defaults to `5.0`
- `output_prefix`: ComfyUI output filename prefix
- `include_images`: when `false` (default), do not embed every decoded frame PNG into the RunPod JSON output; omit this field entirely for the default compact response
- `loras`: optional array of adapters for **compact modes** (`flf2v`, `vace_extend`, `continuous` only—raw `workflow` passthrough ignores it). Omit the field entirely or send `[]` to keep the baked workflow unchanged. Each item uses:
  - `source`: **https** LoRA download URL (any host you trust; the final URL after redirects must stay **https**)
  - `adapter_name`: filename under ComfyUI's `models/loras`, must end in `.safetensors` (use a repo-unique basename per adapter)
  - `adapter_weight`: strength passed to ComfyUI `LoraLoaderModelOnly` (`strength_model`)

Set **`CIVITAI_TOKEN`** in the worker environment when a Civitai-era URL needs auth: if the `source` string contains **`civitai`** (case-insensitive, e.g. `civitai.com` or `civitai.red`) and the URL does not already include a `token` query param, the worker appends `token` (see `handler._download_lora_file`). Optional: `COMFYUI_LORA_DIR` (default `/comfyui/models/loras`), `LORA_DOWNLOAD_MAX_BYTES`, `LORA_DOWNLOAD_TIMEOUT_SEC`.

LoRAs must be compatible with Wan2.2 I2V / VACE; unrelated SDXL/Flux LoRAs may have no effect or break the run.

See `examples/flf2v_request.json` and `examples/continuous_request.json`.

## Seamless Segment Strategy

Generate the first segment with `flf2v` when you have a desired start and end frame. For every following segment, call `vace_extend` with the previous response's `last_frame.data` as the next request's `first_frame`.

When stitching segments client-side, drop the first frame of every segment after segment 0. That avoids rendering the shared boundary frame twice and prevents a visible cut.

The endpoint is segment-based because RunPod serverless is not a live video transport. The response gives you chunked video outputs that can be assembled into a continuous stream.

## Example Async Call

```bash
curl -X POST "https://api.runpod.ai/v2/<endpoint_id>/run" \
  -H "Authorization: Bearer <runpod_api_key>" \
  -H "Content-Type: application/json" \
  -d @examples/flf2v_request.json
```

Poll the returned job ID:

```bash
curl "https://api.runpod.ai/v2/<endpoint_id>/status/<job_id>" \
  -H "Authorization: Bearer <runpod_api_key>"
```

Successful output includes:

- `videos`: generated MP4 files as data URIs
- `last_frame`: final saved chaining frame from the `_last_frame` prefix as a data URI (if available)
- `video_count`: number of video outputs bundled into the payload
- `image_count`: ComfyUI image outputs produced internally (decoded frame PNGs saved by workflows), even when they are omitted from JSON

For debugging oversized responses, temporarily set `"include_images": true` inside `input`. This returns full `images` data URIs, but can blow past RunPod result size limits.

## Raw Workflow Passthrough

Advanced callers can provide a ComfyUI API workflow directly:

```json
{
  "input": {
    "workflow": {
      "6": {
        "inputs": {
          "text": "your prompt"
        },
        "class_type": "CLIPTextEncode"
      }
    }
  }
}
```

When using raw workflows with inputs, include `images` items with `name` and `image`, or use the top-level frame fields. Files are written into ComfyUI's input directory before the workflow is queued. The `loras` request field applies only to compact modes (`flf2v`, `vace_extend`, `continuous`); embedding LoRAs in a raw workflow requires adding `LoraLoaderModelOnly` (or compatible) nodes yourself.

## Static Validation

Run the CPU-only validation before building:

```bash
python scripts/validate_static.py
```

This checks Python syntax and verifies that workflow templates are valid ComfyUI API workflow objects.

## Notes

- The default Dockerfile uses fp8 scaled Wan2.2 weights from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged`.
- The 14B Wan2.2 examples use `wan_2.1_vae.safetensors`; `wan2.2_vae.safetensors` is also baked in for custom workflows.
- If a ComfyUI node package changes its API, export a known-good workflow from ComfyUI and replace the JSON templates while keeping the same request parameter placeholders.
