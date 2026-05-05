import base64
import time
import uuid
from pathlib import Path
from typing import Any

import requests


class ComfyClientError(RuntimeError):
    pass


def _format_workflow_error_summary(prompt_id: str, messages: Any) -> str:
    for item in messages or []:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and item[0] == "execution_error"
            and isinstance(item[1], dict)
        ):
            payload = item[1]
            nid = payload.get("node_id")
            et = payload.get("exception_type")
            em = str(payload.get("exception_message") or "")
            head = em.split("\n", 1)[0][:400]
            return f"ComfyUI workflow failed (prompt_id={prompt_id}, node_id={nid}): {et}: {head}"
    return f"ComfyUI workflow failed (prompt_id={prompt_id})"


class ComfyWorkflowError(ComfyClientError):
    """ComfyUI reported status_str=error; carries raw history status messages for debug output."""

    def __init__(self, prompt_id: str, messages: Any) -> None:
        self.prompt_id = prompt_id
        self.messages = messages
        super().__init__(_format_workflow_error_summary(prompt_id, messages))


class ComfyClient:
    def __init__(self, host: str, port: int, timeout_seconds: int = 1800) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout_seconds = timeout_seconds
        self.client_id = str(uuid.uuid4())

    def wait_until_ready(self, timeout_seconds: int = 120) -> None:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/system_stats", timeout=5)
                if response.ok:
                    return
                last_error = ComfyClientError(f"HTTP {response.status_code}: {response.text[:200]}")
            except requests.RequestException as exc:
                last_error = exc
            time.sleep(1)

        raise ComfyClientError(f"ComfyUI did not become ready: {last_error}")

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow, "client_id": self.client_id},
            timeout=30,
        )
        if not response.ok:
            raise ComfyClientError(f"ComfyUI rejected workflow: {response.status_code} {response.text}")

        data = response.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ComfyClientError(f"ComfyUI response did not include prompt_id: {data}")
        return str(prompt_id)

    def get_history(self, prompt_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
        if not response.ok:
            raise ComfyClientError(f"Could not read ComfyUI history: {response.status_code} {response.text}")
        return response.json()

    def wait_for_prompt(self, prompt_id: str, *, timeout_seconds: int | None = None) -> dict[str, Any]:
        limit = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + limit

        while time.monotonic() < deadline:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                result = history[prompt_id]
                status = result.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages") or []
                    raise ComfyWorkflowError(prompt_id, messages)
                return result
            time.sleep(1)

        raise ComfyClientError(f"ComfyUI prompt timed out after {limit}s")

    def collect_outputs(self, history_result: dict[str, Any], output_dir: Path) -> dict[str, list[dict[str, str]]]:
        outputs: dict[str, list[dict[str, str]]] = {"images": [], "videos": []}

        for node_output in history_result.get("outputs", {}).values():
            for image in node_output.get("images", []):
                item = self._encode_output_file(image, output_dir)
                item["data"] = f"data:image/{item['format']};base64,{item['data']}"
                outputs["images"].append(item)

            for video in node_output.get("videos", []):
                item = self._encode_output_file(video, output_dir)
                item["data"] = f"data:video/{item['format']};base64,{item['data']}"
                outputs["videos"].append(item)

            for animated in node_output.get("animated", []):
                item = self._encode_output_file(animated, output_dir)
                item["data"] = f"data:video/{item['format']};base64,{item['data']}"
                outputs["videos"].append(item)

            for gif in node_output.get("gifs", []):
                item = self._encode_output_file(gif, output_dir)
                item["data"] = f"data:video/{item['format']};base64,{item['data']}"
                outputs["videos"].append(item)

        return outputs

    def _encode_output_file(self, file_info: dict[str, Any], output_dir: Path) -> dict[str, str]:
        filename = str(file_info.get("filename") or "")
        subfolder = str(file_info.get("subfolder") or "")
        if not filename:
            raise ComfyClientError(f"ComfyUI output is missing filename: {file_info}")

        file_path = (output_dir / subfolder / filename).resolve()
        output_root = output_dir.resolve()
        if output_root not in file_path.parents and file_path != output_root:
            raise ComfyClientError(f"Refusing to read output outside {output_root}: {file_path}")
        if not file_path.exists():
            raise ComfyClientError(f"ComfyUI reported missing output file: {file_path}")

        extension = file_path.suffix.lstrip(".").lower() or "octet-stream"
        return {
            "filename": filename,
            "subfolder": subfolder,
            "type": "base64",
            "format": extension,
            "data": base64.b64encode(file_path.read_bytes()).decode("ascii"),
        }
