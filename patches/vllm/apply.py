"""Apply the sparse-MLA patch only to its audited upstream source files."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).resolve().parent
    manifest = json.loads((here / "manifest.json").read_text())
    root = Path(importlib.util.find_spec("vllm").origin).parent.parent

    def verify(stage):
        for name, hashes in manifest["files"].items():
            path = root / name
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != hashes[stage]:
                raise RuntimeError(f"{stage}: unexpected upstream source {path}: {actual}")

    verify("before")
    for name in manifest["patches"]:
        patch = here / name
        cmd = ["patch", "--batch", "--fuzz=0", "-p1", "-d", str(root), "-i", str(patch)]
        subprocess.run([*cmd, "--dry-run"], check=True)
        subprocess.run(cmd, check=True)
    verify("after")
    print("Applied and verified RDNA4 FP8 sparse MLA patch", flush=True)


if __name__ == "__main__":
    main()
