import copy
import json
import random
from pathlib import Path
from typing import Any


WORKFLOW_DIR = Path(__file__).parent / "workflows"


MODEL_DEFAULTS = {
    "i2v_high_model": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "i2v_low_model": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "vace_high_model": "wan2.2_fun_vace_high_noise_14B_fp8_scaled.safetensors",
    "vace_low_model": "wan2.2_fun_vace_low_noise_14B_fp8_scaled.safetensors",
    "text_encoder": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
    "vace_vae": "wan_2.1_vae.safetensors",
}


MODE_TO_TEMPLATE = {
    "flf2v": "wan22_flf2v.json",
    "vace_extend": "wan22_vace_extend.json",
    "continuous": "wan22_vace_extend.json",
}


class WorkflowBuildError(ValueError):
    pass


def build_workflow(request_input: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    if "workflow" in request_input:
        workflow = copy.deepcopy(request_input["workflow"])
        if not isinstance(workflow, dict):
            raise WorkflowBuildError("input.workflow must be an object")
        return workflow

    mode = str(request_input.get("mode", "flf2v"))
    template_name = MODE_TO_TEMPLATE.get(mode)
    if template_name is None:
        raise WorkflowBuildError(f"Unsupported mode '{mode}'. Expected one of: {', '.join(MODE_TO_TEMPLATE)}")

    workflow = _load_template(template_name)
    params = _normalized_params(request_input, files)
    _apply_values(workflow, params)
    return workflow


def _load_template(template_name: str) -> dict[str, Any]:
    template_path = WORKFLOW_DIR / template_name
    with template_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_params(request_input: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    output_prefix = str(request_input.get("output_prefix") or "wan22_segment")
    seed = int(request_input.get("seed", -1))
    if seed < 0:
        seed = random.randint(0, 2**63 - 1)
    steps = int(request_input.get("steps", 20))
    high_noise_end_step = int(request_input.get("high_noise_end_step", max(1, steps // 2)))

    params: dict[str, Any] = {
        **MODEL_DEFAULTS,
        "prompt": str(request_input.get("prompt") or ""),
        "negative_prompt": str(request_input.get("negative_prompt") or ""),
        "width": int(request_input.get("width", 832)),
        "height": int(request_input.get("height", 480)),
        "frames": int(request_input.get("frames", request_input.get("length", 81))),
        "fps": int(request_input.get("fps", 16)),
        "steps": steps,
        "high_noise_end_step": high_noise_end_step,
        "cfg": float(request_input.get("cfg", 5.0)),
        "shift": float(request_input.get("shift", 5.0)),
        "seed": seed,
        "denoise": float(request_input.get("denoise", 1.0)),
        "vace_strength": float(request_input.get("vace_strength", 1.0)),
        "sampler_name": str(request_input.get("sampler_name", "uni_pc")),
        "scheduler": str(request_input.get("scheduler", "simple")),
        "output_prefix": output_prefix,
        "last_frame_prefix": f"{output_prefix}_last_frame",
    }

    for key in MODEL_DEFAULTS:
        if key in request_input:
            params[key] = str(request_input[key])

    if "first_frame" in files:
        params["first_frame"] = files["first_frame"]
        params["start_frame"] = files["first_frame"]
    if "last_frame" in files:
        params["last_frame"] = files["last_frame"]
        params["end_frame"] = files["last_frame"]
    if "source_video" in files:
        params["source_video"] = files["source_video"]
    if "control_image" in files:
        params["control_image"] = files["control_image"]
    if "mask_image" in files:
        params["mask_image"] = files["mask_image"]

    return params


def _apply_values(workflow: dict[str, Any], params: dict[str, Any]) -> None:
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for input_name, input_value in list(inputs.items()):
            if isinstance(input_value, str) and input_value.startswith("${") and input_value.endswith("}"):
                key = input_value[2:-1]
                if key not in params:
                    raise WorkflowBuildError(f"Workflow template requires missing value '{key}'")
                inputs[input_name] = params[key]
