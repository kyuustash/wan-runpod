import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_FILES = [
    ROOT / "comfy_client.py",
    ROOT / "handler.py",
    ROOT / "workflow_builder.py",
]
WORKFLOW_FILES = [
    ROOT / "workflows" / "wan22_flf2v.json",
    ROOT / "workflows" / "wan22_vace_extend.json",
]


def main() -> None:
    for path in PYTHON_FILES:
        py_compile.compile(str(path), doraise=True)

    for path in WORKFLOW_FILES:
        with path.open("r", encoding="utf-8") as handle:
            workflow = json.load(handle)
        if not isinstance(workflow, dict) or not workflow:
            raise ValueError(f"{path} must contain a non-empty API workflow object")
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                raise ValueError(f"{path}:{node_id} must be an object")
            if "class_type" not in node:
                raise ValueError(f"{path}:{node_id} is missing class_type")
            if "inputs" not in node:
                raise ValueError(f"{path}:{node_id} is missing inputs")

    print("Static validation passed")


if __name__ == "__main__":
    main()
