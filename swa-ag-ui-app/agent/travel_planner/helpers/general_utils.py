from pathlib import Path
from typing import Any

import yaml


def read_yaml(yaml_path: Path) -> dict[str, Any]:
    with open(yaml_path, encoding="utf-8") as file:
        prompt_dict: dict[str, Any] = yaml.safe_load(file)
    return prompt_dict
