import base64
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse, urlencode, urlunparse

import requests
import runpod

from comfy_client import ComfyClient, ComfyClientError, ComfyWorkflowError
from workflow_builder import WorkflowBuildError, build_workflow


COMFYUI_DIR = Path(os.getenv("COMFYUI_DIR", "/comfyui"))
COMFYUI_INPUT_DIR = Path(os.getenv("COMFYUI_INPUT_DIR", str(COMFYUI_DIR / "input")))
COMFYUI_OUTPUT_DIR = Path(os.getenv("COMFYUI_OUTPUT_DIR", str(COMFYUI_DIR / "output")))
COMFYUI_LORA_DIR = Path(os.getenv("COMFYUI_LORA_DIR", str(COMFYUI_DIR / "models" / "loras")))
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = int(os.getenv("COMFYUI_PORT", "8188"))
COMFYUI_TIMEOUT_SECONDS = int(os.getenv("COMFYUI_TIMEOUT_SECONDS", "1800"))
LORA_DOWNLOAD_MAX_BYTES = int(os.getenv("LORA_DOWNLOAD_MAX_BYTES", str(2 * 1024 * 1024 * 1024)))
LORA_DOWNLOAD_TIMEOUT_SEC = int(os.getenv("LORA_DOWNLOAD_TIMEOUT_SEC", "600"))
_COMFY_PROCESS: subprocess.Popen[bytes] | None = None
_COMFY_CLIENT: ComfyClient | None = None
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SAFETENSORS_HEADER_MAX_BYTES = 1_000_000
_DEBUG_MAX_TRACEBACK_LINES = 40
_DEBUG_MAX_TRACEBACK_LINE_LEN = 600
_LORA_DEBUG_PREFIX_BYTES = 16


class HandlerError(ValueError):
    pass


def handler(event: dict[str, Any]) -> dict[str, Any]:
    raw_input = event.get("input")
    debug_logs = isinstance(raw_input, dict) and bool(raw_input.get("debug_logs", False))
    lora_debug: list[dict[str, Any]] = []
    workflow: dict[str, Any] | None = None

    try:
        if not isinstance(raw_input, dict):
            raise HandlerError("event.input must be an object")

        request_input = raw_input
        client = ensure_comfy_client()
        saved_files = save_request_files(request_input)
        resolve_and_download_loras(
            request_input,
            saved_files,
            lora_debug if debug_logs else None,
        )
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
    except Exception as exc:
        return _build_error_response(
            exc,
            debug_logs,
            workflow=workflow,
            lora_debug=lora_debug,
        )


def _build_error_response(
    exc: Exception,
    debug_logs: bool,
    *,
    workflow: dict[str, Any] | None,
    lora_debug: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(exc, (ComfyClientError, HandlerError, WorkflowBuildError)):
        msg = str(exc)
    else:
        msg = f"Unhandled worker error: {exc}"

    body: dict[str, Any] = {"status": "error", "error": msg}
    if not debug_logs:
        return body

    debug: dict[str, Any] = {
        "exception_type": type(exc).__name__,
        "lora_dir": str(COMFYUI_LORA_DIR.resolve()),
    }
    if isinstance(exc, ComfyWorkflowError):
        debug["prompt_id"] = exc.prompt_id
        debug["comfy_messages"] = _cap_comfy_messages(exc.messages)
    if lora_debug:
        debug["loras"] = lora_debug
    if workflow:
        lora_nodes = _lora_loader_nodes_from_workflow(workflow)
        if lora_nodes:
            debug["lora_loader_nodes"] = lora_nodes

    body["debug"] = debug
    return body


def _cap_comfy_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    capped: list[Any] = []
    for raw in messages:
        if not (isinstance(raw, (list, tuple)) and len(raw) >= 2):
            capped.append(raw)
            continue
        ev, payload = raw[0], raw[1]
        if ev == "execution_error" and isinstance(payload, dict):
            pl = dict(payload)
            tb = pl.get("traceback")
            if isinstance(tb, list):
                pl["traceback"] = [
                    str(line)[:_DEBUG_MAX_TRACEBACK_LINE_LEN] for line in tb[:_DEBUG_MAX_TRACEBACK_LINES]
                ]
            capped.append([ev, pl])
        else:
            capped.append([ev, payload])
    return capped


def _sanitize_url_for_logs(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _lora_file_debug_prefix_hex(path: Path, max_bytes: int = _LORA_DEBUG_PREFIX_BYTES) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
    except OSError:
        return ""
    return data.hex()


def _lora_loader_nodes_from_workflow(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or node.get("class_type") != "LoraLoaderModelOnly":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        found.append(
            {
                "node_id": str(node_id),
                "lora_name": inputs.get("lora_name"),
                "strength_model": inputs.get("strength_model"),
            }
        )
    return sorted(found, key=lambda x: x["node_id"])


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


def save_request_files(request_input: dict[str, Any]) -> dict[str, Any]:
    request_id = _safe_filename(str(request_input.get("request_id") or str(int(time.time() * 1000))))
    saved: dict[str, Any] = {}

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


def resolve_and_download_loras(
    request_input: dict[str, Any],
    saved_files: dict[str, Any],
    debug_lora_records: list[dict[str, Any]] | None = None,
) -> None:
    """Populate saved_files[\"loras\"] when input.loras is non-empty; download each file to COMFYUI_LORA_DIR."""
    raw = request_input.get("loras")
    if raw is None or raw == []:
        return
    if not isinstance(raw, list):
        raise HandlerError("input.loras must be an array when provided")

    specs: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    COMFYUI_LORA_DIR.mkdir(parents=True, exist_ok=True)

    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise HandlerError(f"input.loras[{idx}] must be an object")

        url = item.get("source") or item.get("url")
        adapter_name = item.get("adapter_name")
        weight_any = item.get("adapter_weight", 1.0)

        if not isinstance(url, str) or not url.strip():
            raise HandlerError(f"input.loras[{idx}] must include a non-empty string source")
        if not isinstance(adapter_name, str) or not adapter_name.strip():
            raise HandlerError(f"input.loras[{idx}] must include a non-empty string adapter_name")
        safe_name = _safe_filename(adapter_name)
        if not safe_name.lower().endswith(".safetensors"):
            raise HandlerError(f"input.loras[{idx}] adapter_name must end with .safetensors")

        try:
            weight = float(weight_any)
        except (TypeError, ValueError) as exc:
            raise HandlerError(f"input.loras[{idx}] adapter_weight must be a number") from exc

        if safe_name in seen_names:
            raise HandlerError(f"Duplicate adapter_name in loras list: {safe_name}")
        seen_names.add(safe_name)

        _validate_https_lora_url(url)
        dest = COMFYUI_LORA_DIR / safe_name
        used_cache = dest.exists() and dest.stat().st_size > 0 and _is_valid_safetensors_file(dest)
        content_type: str | None = None
        final_url: str | None = None

        if used_cache:
            pass
        else:
            if dest.exists():
                dest.unlink()
            content_type, final_url = _download_lora_file(url, dest)
            if not _is_valid_safetensors_file(dest):
                if dest.exists():
                    dest.unlink()
                hint = ""
                if content_type and "text/html" in content_type.lower():
                    hint = (
                        " Server returned text/html; use a direct https file URL "
                        "(Hugging Face /resolve/…/file.safetensors) or set CIVITAI_TOKEN if needed."
                    )
                raise HandlerError(
                    f"LoRA download for {safe_name} is not valid safetensors; "
                    f"URL may return HTML or require auth.{hint}"
                )

        if debug_lora_records is not None:
            entry: dict[str, Any] = {
                "adapter_name": safe_name,
                "path": str(dest.resolve()),
                "size_bytes": dest.stat().st_size,
                "first_16_bytes_hex": _lora_file_debug_prefix_hex(dest),
                "safetensors_header_ok": _is_valid_safetensors_file(dest),
                "used_cached_file": used_cache,
            }
            if not used_cache and final_url:
                entry["download_content_type"] = content_type
                entry["download_final_url"] = _sanitize_url_for_logs(final_url)
            debug_lora_records.append(entry)

        specs.append({"adapter_name": safe_name, "adapter_weight": weight})

    saved_files["loras"] = specs


def _validate_https_lora_url(url: str) -> None:
    parsed = urlparse(url)
    if (parsed.scheme or "").lower() != "https":
        raise HandlerError("LoRA source must use https URLs")
    if not parsed.hostname:
        raise HandlerError("LoRA source URL must include a host")


def _query_has_token_param(query: str) -> bool:
    return any(k.lower() == "token" for k in parse_qs(query, keep_blank_values=True))


def _is_valid_safetensors_file(path: Path) -> bool:
    """True if path looks like a safetensors file (parsable JSON header after 8-byte LE length)."""
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if size < 10:
        return False
    try:
        with path.open("rb") as handle:
            length_prefix = handle.read(8)
            if len(length_prefix) != 8:
                return False
            (header_size,) = struct.unpack("<Q", length_prefix)
            if header_size < 2 or header_size > _SAFETENSORS_HEADER_MAX_BYTES:
                return False
            if size < 8 + header_size:
                return False
            header_blob = handle.read(header_size)
            if len(header_blob) != header_size:
                return False
            json.loads(header_blob.decode("utf-8"))
    except (OSError, struct.error, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return False
    return True


def _download_lora_file(url: str, dest: Path) -> tuple[str | None, str]:
    token = os.environ.get("CIVITAI_TOKEN")
    download_url = url
    parsed = urlparse(url)
    if (
        token
        and "civitai" in url.lower()
        and not _query_has_token_param(parsed.query)
    ):
        sep = "&" if "?" in url else "?"
        download_url = f"{url}{sep}{urlencode({'token': token})}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/octet-stream,*/*",
    }

    dest_part = dest.with_suffix(dest.suffix + ".download")
    if dest_part.exists():
        dest_part.unlink()

    try:
        with requests.get(
            download_url,
            headers=headers,
            stream=True,
            timeout=LORA_DOWNLOAD_TIMEOUT_SEC,
            allow_redirects=True,
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type")
            final = urlparse(response.url)
            if (final.scheme or "").lower() != "https":
                raise HandlerError(f"Redirect left non-https URL for LoRA download: {response.url}")
            if not final.hostname:
                raise HandlerError(f"Redirect left URL without host for LoRA download: {response.url}")

            content_length_header = response.headers.get("Content-Length")
            if content_length_header and content_length_header.isdigit():
                clen = int(content_length_header)
                if clen > LORA_DOWNLOAD_MAX_BYTES:
                    raise HandlerError(
                        "LoRA download Content-Length exceeds LORA_DOWNLOAD_MAX_BYTES "
                        f"({LORA_DOWNLOAD_MAX_BYTES})"
                    )

            written = 0
            dest_part.parent.mkdir(parents=True, exist_ok=True)
            with dest_part.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > LORA_DOWNLOAD_MAX_BYTES:
                        raise HandlerError(
                            f"LoRA download exceeded LORA_DOWNLOAD_MAX_BYTES ({LORA_DOWNLOAD_MAX_BYTES})"
                        )
                    handle.write(chunk)

        dest_part.replace(dest)
        return content_type, str(response.url)
    except Exception:
        if dest_part.exists():
            dest_part.unlink()
        raise


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
