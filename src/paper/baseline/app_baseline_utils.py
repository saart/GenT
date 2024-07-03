import json
import os
from typing import Dict, List

METADATA_DIR = os.path.join(os.path.dirname(__file__), "work_folder", "app")
METADATA_FILE = os.path.join(METADATA_DIR, "metadata.json")
EMPTY = "Unknown"


global_metadata: Dict[str, List[str]] = {}


def remember_component_fields(component_name: str, field_names: List[str]) -> None:
    global_metadata[component_name] = field_names


def get_component_fields(component_name: str) -> List[str]:
    return global_metadata[component_name]


def store_global_metadata() -> None:
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(METADATA_FILE, "w") as f:
        json.dump(global_metadata, f)


def load_global_metadata() -> None:
    global global_metadata
    with open(METADATA_FILE, "r") as f:
        global_metadata = json.load(f)
