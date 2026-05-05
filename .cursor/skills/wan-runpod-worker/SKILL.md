---
name: wan-runpod-worker
description: Work on the Wan2.2 RunPod ComfyUI serverless worker. Use when deploying, debugging, editing ComfyUI workflows, changing the handler API, explaining requests/responses, or working on FLF2V, VACE, or continuous video generation.
---

# Wan RunPod Worker

## Quick Context

This project packages a RunPod serverless worker around ComfyUI for Wan2.2 video generation.

- `Dockerfile`: base image, custom nodes, baked Wan2.2 fp8 model downloads.
- `handler.py`: RunPod handler, base64 input decoding, ComfyUI startup, response shape.
- `workflow_builder.py`: compact API to ComfyUI workflow template injection.
- `comfy_client.py`: ComfyUI queue, history polling, output encoding.
- `workflows/wan22_flf2v.json`: first/last-frame segment workflow.
- `workflows/wan22_vace_extend.json`: VACE continuation workflow.

## Before Editing

1. Read the files related to the task instead of relying on memory.
2. Keep workflow JSON in ComfyUI API format: node-id object map with `class_type` and `inputs`.
3. Preserve `${placeholder}` template values unless intentionally changing the compact API.
4. Do not move models to a network-volume design unless the user asks; this repo bakes models into the image.
5. When adding a new workflow capability, make the current build self-contained. Do not rely on a later/manual ComfyUI install step.

## Dependency Checklist

For every new model, LoRA, node, or package, update the same change set:

- Hugging Face model files: add `comfy model download` lines in `Dockerfile` with the correct `--relative-path` and `--filename`.
- LoRAs: place under `models/loras`, add loader nodes in workflow templates, and add request/model defaults in `workflow_builder.py` if caller-selectable.
- Custom ComfyUI nodes: install with `comfy-node-install` or a pinned `git clone` in `Dockerfile`; include any node-specific Python/system dependencies.
- Python runtime packages: add to `requirements.txt`; avoid relying on packages that happen to exist in the base image unless they are part of the documented base contract.
- Workflow references: make sure filenames in `workflows/*.json` match the baked filenames exactly.
- Documentation/examples: update `README.md` or `examples/` when request fields or dependency expectations change.

Prefer pinned Hugging Face revisions or immutable URLs for production-sensitive assets. If using `main`, mention that rebuilds can pull changed weights.

## API Conventions

Supported modes:

- `flf2v`: requires `first_frame` and `last_frame`; returns MP4 video plus `last_frame` with compact JSON by default.
- `vace_extend`: uses the previous segment's `last_frame.data` as `first_frame`.
- `continuous`: alias for `vace_extend`.
- raw `workflow`: caller provides a ComfyUI API workflow directly.

Set `input.include_images: true` temporarily when full decoded PNG stacks must be surfaced; omit it normally to keep RunPod result payloads tiny.

Continuity rule: chain `last_frame.data` into the next request and drop the duplicate first frame when stitching later segments.

Optional `input.comfyui_timeout_seconds` (integer 1–86400) overrides env `COMFYUI_TIMEOUT_SECONDS` for that job only; successful responses include `comfyui_timeout_seconds` with the effective wait cap.

## Output Troubleshooting

If RunPod shows `COMPLETED` without `output`, or logs `Failed to return job results | 400 ... /job-done/...` after `Prompt executed`, assume the returned JSON payload is too large before suspecting generation failure.

Response-size guardrails:

- Keep `include_images` omitted or `false` for normal jobs.
- Expect workflows to produce many internal PNG frames; `image_count` reports this without returning them.
- Test oversized-output fixes with `frames: 9`, then normal `frames: 81`.
- Prefer object storage URLs if videos become too large for base64 RunPod results.

## Model Defaults

Model defaults live in `workflow_builder.py`:

- Wan2.2 I2V fp8 high/low noise models.
- Wan2.2 VACE fp8 high/low noise models.
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors`.
- `wan_2.1_vae.safetensors` for 14B templates; `wan2.2_vae.safetensors` is also baked for custom workflows.

## Validation

After Python or workflow-template edits, run:

```bash
python scripts/validate_static.py
```

Use this as a syntax/template check only. Real ComfyUI node compatibility and generation quality require a RunPod GPU smoke test.

## Deployment Notes

Build for RunPod with (prefer `--build-arg HF_TOKEN` for cold model pulls; see `README.md` **RunPod builder timeouts**):

```bash
docker build \
  --build-arg HF_TOKEN=hf_... \
  --platform linux/amd64 \
  -t <registry>/<image>:wan22 .
docker push <registry>/<image>:wan22
```

For endpoint docs, request examples, and response shape, check `README.md` and `examples/`.
