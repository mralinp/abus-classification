"""Push a kernel: python kaggle/push.py <name>

Committed metadata names sources without an owner. Owners come from ~/.kaggle/owners.json, kept out of the repo:
    {"tdsc-abus2023-trainval": "<owner>", "tdsc-abus2023-test": "<owner>"}
A source not listed there belongs to the kernel's own account. Uses the CLI's current account (or KAGGLE_API_TOKEN).
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def resolve(meta, owners):
    """Metadata with every source as owner/slug; the default owner is the one in the kernel's id."""
    default = meta["id"].split("/")[0]
    full = lambda slug: f"{owners.get(slug, default)}/{slug}"
    return {**meta, "dataset_sources": [full(s) for s in meta["dataset_sources"]],
            "kernel_sources": [full(s) for s in meta["kernel_sources"]]}


if __name__ == "__main__":
    source = Path(__file__).parent / sys.argv[1]
    config = Path.home() / ".kaggle" / "owners.json"
    owners = json.loads(config.read_text()) if config.exists() else {}
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(source, tmp, dirs_exist_ok=True)
        meta = resolve(json.loads((source / "kernel-metadata.json").read_text()), owners)
        (Path(tmp) / "kernel-metadata.json").write_text(json.dumps(meta))
        subprocess.run(["kaggle", "kernels", "push", "-p", tmp], check=True)
