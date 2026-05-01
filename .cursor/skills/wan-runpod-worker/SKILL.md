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

## API Conventions

Supported modes:

- `flf2v`: requires `first_frame` and `last_frame`; returns an MP4 and `last_frame`.
- `vace_extend`: uses the previous segment's `last_frame.data` as `first_frame`.
- `continuous`: alias for `vace_extend`.
- raw `workflow`: caller provides a ComfyUI API workflow directly.

Continuity rule: chain `last_frame.data` into the next request and drop the duplicate first frame when stitching later segments.

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

Build for RunPod with:

```bash
docker build --platform linux/amd64 -t <registry>/<image>:wan22 .
docker push <registry>/<image>:wan22
```

For endpoint docs, request examples, and response shape, check `README.md` and `examples/`.
