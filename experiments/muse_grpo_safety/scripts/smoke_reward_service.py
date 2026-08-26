#!/usr/bin/env python3
"""Start one official reward service, exercise it, and save a receipt."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("service", choices=("musecritic", "mucodec"))
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--rubric-jsonl", type=Path)
    parser.add_argument("--tokens-jsonl", type=Path)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--num-steps", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _wait_for_health(url: str, process: subprocess.Popen, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "service did not answer"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup with code {process.returncode}")
        try:
            health = _request_json(url, timeout=5.0)
            if health.get("status") == "healthy" and health.get("model_loaded"):
                return health
            last_error = f"unhealthy response: {health}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(2.0)
    raise TimeoutError(f"service health timeout: {last_error}")


def _load_rubric(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        for message in row.get("messages", []):
            if message.get("role") == "user" and message.get("content"):
                return str(message["content"])
    raise ValueError(f"no user rubric found in {path}")


def _load_tokens(path: Path, row_index: int) -> list[int]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    tokens = rows[row_index].get("audio_token_ids") or []
    if not tokens:
        raise ValueError(f"row {row_index} has no audio_token_ids")
    return [int(value) for value in tokens]


def _command(args: argparse.Namespace, port: int) -> list[str]:
    command = [
        str(args.python.resolve()),
        str(args.script.resolve()),
        "--model",
        str(args.model.resolve()),
        "--device",
        "cuda:0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if args.service == "musecritic":
        command.extend(["--max-new-tokens", str(args.max_new_tokens)])
    return command


def main() -> None:
    args = parse_args()
    port = args.port or (8002 if args.service == "musecritic" else 8003)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    receipt: dict[str, Any] = {
        "schema": "vocal2accomp_muse.reward_service_smoke",
        "service": args.service,
        "status": "FAIL",
        "python": str(args.python.resolve()),
        "script": str(args.script.resolve()),
        "model": str(args.model.resolve()),
        "port": port,
    }
    process = None
    with args.log.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                _command(args, port),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONNOUSERSITE": "1"},
                start_new_session=True,
            )
            health = _wait_for_health(
                f"http://127.0.0.1:{port}/health",
                process,
                args.startup_timeout,
            )
            ready_at = time.monotonic()
            if args.service == "musecritic":
                if args.audio is None or args.rubric_jsonl is None:
                    raise ValueError("MuseCritic requires --audio and --rubric-jsonl")
                response = _request_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    payload={
                        "model": "MuseCritic",
                        "messages": [{
                            "role": "user",
                            "content": _load_rubric(args.rubric_jsonl.resolve()),
                        }],
                        "audios": [str(args.audio.resolve())],
                        "max_tokens": args.max_new_tokens,
                        "temperature": 0,
                    },
                    timeout=args.request_timeout,
                )
                content = response["choices"][0]["message"]["content"]
                result = json.loads(content)
                receipt["result"] = {
                    "reward_scores": result.get("reward_scores", {}),
                    "critique_chars": len(result.get("infer_critic", "")),
                    "audio": str(args.audio.resolve()),
                }
            else:
                if args.tokens_jsonl is None:
                    raise ValueError("MuCodec requires --tokens-jsonl")
                tokens = _load_tokens(args.tokens_jsonl.resolve(), args.row)
                response = _request_json(
                    f"http://127.0.0.1:{port}/v1/audio/decode",
                    payload={
                        "tokens": tokens,
                        "duration": args.duration,
                        "num_steps": args.num_steps,
                    },
                    timeout=args.request_timeout,
                )
                receipt["result"] = {
                    key: response[key]
                    for key in (
                        "sample_rate",
                        "duration",
                        "channels",
                        "token_count",
                        "inference_time",
                    )
                }
                receipt["result"]["audio_base64_chars"] = len(response["audio_base64"])
            receipt.update(
                {
                    "status": "PASS",
                    "health": health,
                    "startup_seconds": ready_at - started,
                    "request_seconds": time.monotonic() - ready_at,
                }
            )
        except Exception as exc:
            receipt["error"] = repr(exc)
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            receipt["service_returncode"] = None if process is None else process.returncode
            receipt["elapsed_seconds"] = time.monotonic() - started

    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
