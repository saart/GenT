from typing import Union, Tuple

import os

import json

RESULTS_FILE = os.path.join(os.path.dirname(__file__), "baseline_results.json")

def store_result(result_key: str, result_value: Union[float, Tuple[float, float]]) -> None:
    if not os.path.exists(RESULTS_FILE):
        existing_results = {}
    else:
        with open(RESULTS_FILE, "r") as f:
            existing_results = json.load(f)
    existing_results[result_key.lower()] = result_value
    with open(RESULTS_FILE, "w") as f:
        json.dump(existing_results, f, indent=4)


def load_results(result_key: str):
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Results file {RESULTS_FILE} not found")
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)
    return results[result_key]
