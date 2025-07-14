# trader_assist/engine/state_store.py

import json
import os

DATA_DIR = "data"

def _get_path(filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)

def load_json(filename):
    path = _get_path(filename)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def save_json(filename, data):
    path = _get_path(filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)