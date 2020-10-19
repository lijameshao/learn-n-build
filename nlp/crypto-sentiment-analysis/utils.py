import json
from pathlib import Path


def json_save(data, fp):
    file_dir = Path(fp).parents[0]
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    with open(fp, "w") as f:
        json.dump(data, f)
    return

def json_load(fp):
    with open(fp, "r") as f:
        data = json.load(f)
    return data

