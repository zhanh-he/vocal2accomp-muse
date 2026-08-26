#!/usr/bin/env python3
"""Qualify staged Muse GRPO assets without allocating a GPU."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path


IMPORTS = {
    "torch": "torch",
    "torchaudio": "torchaudio",
    "transformers": "transformers",
    "swift": "ms-swift",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "soundfile": "soundfile",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--muse-model", type=Path, required=True)
    parser.add_argument("--musecritic-model", type=Path, required=True)
    parser.add_argument("--mucodec-root", type=Path, required=True)
    parser.add_argument("--musecritic-service", type=Path, required=True)
    parser.add_argument("--mucodec-service", type=Path, required=True)
    parser.add_argument(
        "--musecritic-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument(
        "--mucodec-python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _file_record(path: Path) -> dict[str, object]:
    exists = path.is_file()
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
    }


def _service_help(path: Path, python: Path) -> dict[str, object]:
    if not path.is_file():
        return {
            "path": str(path),
            "python": str(python),
            "ok": False,
            "error": "missing script",
        }
    if not python.is_file():
        return {
            "path": str(path),
            "python": str(python),
            "ok": False,
            "error": "missing Python interpreter",
        }
    python_home = python.resolve().parent.parent
    library_path = str(python_home / "lib")
    if os.environ.get("LD_LIBRARY_PATH"):
        library_path += f":{os.environ['LD_LIBRARY_PATH']}"
    process = subprocess.run(
        [str(python), str(path), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "LD_LIBRARY_PATH": library_path},
    )
    return {
        "path": str(path),
        "python": str(python),
        "ok": process.returncode == 0,
        "returncode": process.returncode,
        "stderr_tail": process.stderr[-2000:],
    }


def main() -> None:
    args = parse_args()
    imports = {}
    for module_name, distribution_name in IMPORTS.items():
        try:
            importlib.import_module(module_name)
            imports[module_name] = {
                "ok": True,
                "version": importlib.metadata.version(distribution_name),
            }
        except Exception as exc:  # qualification receipt must preserve failures
            imports[module_name] = {"ok": False, "error": repr(exc)}

    files = {
        "muse": _file_record(args.muse_model / "model.safetensors"),
        "musecritic_index": _file_record(
            args.musecritic_model / "model.safetensors.index.json"
        ),
        "musecritic_shard_1": _file_record(
            args.musecritic_model / "model-00001-of-00004.safetensors"
        ),
        "mucodec": _file_record(args.mucodec_root / "weights" / "mucodec.pt"),
        "muq": _file_record(args.mucodec_root / "muq_dev" / "muq.pt"),
        "audioldm": _file_record(args.mucodec_root / "tools" / "audioldm_48k.pth"),
    }
    services = {
        "musecritic": _service_help(
            args.musecritic_service,
            args.musecritic_python,
        ),
        "mucodec": _service_help(
            args.mucodec_service,
            args.mucodec_python,
        ),
    }
    ok = (
        all(record["ok"] for record in imports.values())
        and all(record["exists"] for record in files.values())
        and all(record["ok"] for record in services.values())
    )
    receipt = {
        "schema": "vocal2accomp_muse.grpo_stack_qualification",
        "ok": ok,
        "python": sys.version,
        "imports": imports,
        "files": files,
        "service_cli": services,
    }
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
