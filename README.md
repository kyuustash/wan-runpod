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

Build for RunPod's Linux AMD64 runtime:

```bash
docker build --platform linux/amd64 -t <registry>/<image>:wan22 .
docker push <registry>/<image>:wan22
```

The baked model set is large. Set a large enough container disk in the RunPod template, and use a high-VRAM GPU class suitable for Wan2.2 14B fp8 video workflows.

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

When using raw workflows with inputs, include `images` items with `name` and `image`, or use the top-level frame fields. Files are written into ComfyUI's input directory before the workflow is queued.

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
