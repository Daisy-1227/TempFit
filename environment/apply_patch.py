"""Apply a checked source patch; refuses unexpected or already changed input."""

from pathlib import Path
import difflib, hashlib, importlib.util, json, re


def main():
    root = Path(__file__).parent / "patches"
    record = json.loads((root / "x-transformers-2.3.12.json").read_text())
    spec = importlib.util.find_spec("x_transformers")
    target = Path(spec.origin).parent / "x_transformers.py"
    digest = lambda x: hashlib.sha256(x).hexdigest()
    data = target.read_bytes()
    if digest(data) == record["patched_sha256"]:
        print("Verified patch already applied")
        return
    if digest(data) != record["original_sha256"]:
        raise ValueError("Unexpected x-transformers source; refusing to patch")
    patch = (root / "x-transformers-2.3.12.patch").read_bytes()
    if digest(patch) != record["patch_sha256"]:
        raise ValueError("Patch hash mismatch")
    original = data.decode().splitlines(True)
    result = []
    cursor = 0
    lines = patch.decode().splitlines(True)
    i = 2
    while i < len(lines):
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", lines[i])
        if not m:
            raise ValueError("Invalid unified patch")
        start = int(m.group(1)) - 1
        result.extend(original[cursor:start])
        cursor = start
        i += 1
        while i < len(lines) and not lines[i].startswith("@@"):
            kind = lines[i][0]
            body = lines[i][1:]
            if kind in (" ", "-"):
                if original[cursor] != body:
                    raise ValueError("Patch context mismatch")
                cursor += 1
            if kind in (" ", "+"):
                result.append(body)
            i += 1
    result.extend(original[cursor:])
    new = "".join(result).encode()
    if digest(new) != record["patched_sha256"]:
        raise ValueError("Patched source hash mismatch")
    target.write_bytes(new)
    print("Verified x-transformers patch applied")


if __name__ == "__main__":
    main()
