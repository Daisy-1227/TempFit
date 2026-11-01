import json

from .io import save


def log_update(output, row):
    with (output / "metrics.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    save(output / "status.json", dict(status="running", **row))
