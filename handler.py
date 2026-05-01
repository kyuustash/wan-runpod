import base64
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import runpod

from comfy_client import ComfyClient, ComfyClientError
from workflow_builder import WorkflowBuildError, build_workflow


COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/comfyui"))
COMFYUI_INPUT_DIR = Path(os.getenv("COMFYUI_INPUT_DIR", str(COMFYUI_DIR / "input")))
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", str(COMFYUI_DIR / "output")))
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "1800"))

_COMFY_PROCESS: subprocess.Popen[bytes] | None = None
_COMFY_CLIENT: ComfyClient | None = None
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


class HandlerError(ValueError):
    pass


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        request_input = event.get("input") or {}
        if not isinstance(request_input, dict):
            raise HandlerError("event.input must be an object")

        client = ensure_comfy_client()
        saved_files = save_request_files(request_input)
        workflow = build_workflow(request_input, saved_files)

        prompt_id = client.queue_prompt(workflow)
        history = client.wait_for_prompt(prompt_id)
        outputs = client.collect_outputs(history, COMFYUI_OUTPUT_DIR)

        videos = outputs["videos"]
        images = outputs["images"]

        response = {
            "status": "success",
            "prompt_id": prompt_id,
            "mode": request_input.get("mode", "raw" if "workflow" in request_input else "flf2v"),
            "videos": videos,
            "last_frame": _select_last_frame(images),
            "video_count": len(videos),
            "image_count": len(images),
        }

        include_images = bool(request_input.get("include_images", False))
        if include_images:
            response["images"] = images

        return response
    except (ComfyClientError, HandlerError, WorkflowBuildError) as exc:
        return {"status": "error", "error": str(exc)}
    except Exception as exc:  # RunPod should receive structured errors even for unexpected failures.
        return {"status": "error", "error": f"Unhandled worker error: {exc}"}


def ensure_comfy_client() -> ComfyClient:
    global _COMFY_CLIENT

    if _COMFY_CLIENT is None:
        start_comfyui()
        _COMFY_CLIENT = ComfyClient(COMFYUI_HOST, COMFYUI_PORT, COMFYUI_TIMEOUT_SECONDS)
        _COMFY_CLIENT.wait_until_ready()

    return _COMFY_CLIENT


def start_comfyui() -> None:
    global _COMFY_PROCESS

    if _COMFY_PROCESS is not None and _COMFY_PROCESS.poll() is None:
        return

    COMFYUI_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    COMFYUI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        "python",
        str(COMFYUI_DIR / "main.py"),
        "--listen",
        COMFYUI_HOST,
        "--port",
        str(COMFYUI_PORT),
    ]
    extra_args = os.getenv("COMFYUI_EXTRA_ARGS", "").split()
    command.extend(extra_args)

    _COMFY_PROCESS = subprocess.Popen(command, cwd=str(COMFYUI_DIR))
    time.sleep(1)


def save_request_files(request_input: dict[str, Any]) -> dict[str, str]:
    request_id = _safe_filename(str(request_input.get("request_id") or str(int(time.time() * 1000))))
    saved: dict[str, str] = {}

    file_fields = {
        "first_frame": "first_frame.png",
        "last_frame": "last_frame.png",
        "control_image": "control_image.png",
        "mask_image": "mask_image.png",
        "source_video": "source_video.mp4",
    }

    for field, default_name in file_fields.items():
        encoded = request_input.get(field)
        if encoded:
            filename = f"{request_id}_{default_name}"
            _write_base64_file(encoded, COMFYUI_INPUT_DIR / filename)
            saved[field] = filename

    for image in request_input.get("images", []) or []:
        if not isinstance(image, dict) or "name" not in image or "image" not in image:
            raise HandlerError("Each input.images item must include name and image")
        filename = _safe_filename(str(image["name"]))
        _write_base64_file(image["image"], COMFYUI_INPUT_DIR / filename)
        saved[Path(filename).stem] = filename

    return saved


def _write_base64_file(encoded: Any, path: Path) -> None:
    if not isinstance(encoded, str):
        raise HandlerError(f"File payload for {path.name} must be a base64 string")
    payload = encoded.split(",", 1)[1] if "," in encoded else encoded
    path.write_bytes(base64.b64decode(payload, validate=True))


def _safe_filename(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name).strip("._")
    if not cleaned:
        raise HandlerError("Filename cannot be empty")
    return cleaned[:120]


def _select_last_frame(images: list[dict[str, str]]) -> dict[str, str] | None:
    preferred = [image for image in images if "last_frame" in image.get("filename", "")]
    if preferred:
        return preferred[-1]
    return images[-1] if images else None


runpod.serverless.start({"handler": handler})
