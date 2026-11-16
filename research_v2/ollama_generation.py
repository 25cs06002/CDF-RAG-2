"""Memory-bounded, observable Ollama generation for B1-B6.

Keep benchmark evidence separate from generation failure; retry ONLY transient
transport/server failures and never turn exceptions into fabricated answers.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import requests


class OllamaGenerationError(RuntimeError):
    """A generation attempt failed; caller retains retrieved evidence."""


def _int_setting(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def generation_settings() -> dict:
    """Version these parameters alongside the run for reproducible comparison."""
    return {
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2:latest"),
        "endpoint": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        "num_ctx": _int_setting("OLLAMA_NUM_CTX", 2048, minimum=256),
        "num_predict": _int_setting("OLLAMA_NUM_PREDICT", 256, minimum=1),
        "unload_every_n": _int_setting("OLLAMA_UNLOAD_EVERY_N", 5, minimum=1),
        "attempts": _int_setting("OLLAMA_GENERATION_ATTEMPTS", 3, minimum=1),
        "timeout_seconds": _int_setting("OLLAMA_REQUEST_TIMEOUT", 180, minimum=1),
        "keep_alive": os.environ.get("OLLAMA_KEEP_ALIVE_REQUEST", "5m"),
        "temperature": 0,
    }


def check_ollama(settings: dict | None = None) -> dict:
    cfg = settings or generation_settings()
    url = cfg["endpoint"] + "/api/tags"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        names = {item["name"] for item in response.json().get("models", [])}
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        raise OllamaGenerationError(
            "Ollama not available on configured local endpoint: " + type(exc).__name__
        ) from exc
    if cfg["model"] not in names:
        raise OllamaGenerationError(
            f"Ollama model {cfg['model']!r} not installed; available: {sorted(names)}"
        )
    return {"endpoint": cfg["endpoint"], "model": cfg["model"], "status": "ready"}


def generate(prompt: str, *, unload_after: bool = False, settings: dict | None = None) -> str:
    cfg = settings or generation_settings()
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "stream": False,
        "keep_alive": 0 if unload_after else cfg["keep_alive"],
        "options": {
            "temperature": cfg["temperature"],
            "num_predict": cfg["num_predict"],
            "num_ctx": cfg["num_ctx"],
        },
    }
    last_error = None
    for attempt in range(cfg["attempts"]):
        try:
            response = requests.post(
                cfg["endpoint"] + "/api/generate",
                json=payload,
                timeout=cfg["timeout_seconds"],
            )
            response.raise_for_status()
            answer = response.json().get("response")
            if not isinstance(answer, str) or not answer.strip():
                raise OllamaGenerationError("empty_generation_response")
            return answer
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = exc
        except requests.exceptions.HTTPError as exc:
            status = getattr(exc.response, "status_code", None)
            if status not in {429, 500, 502, 503, 504}:
                raise OllamaGenerationError(f"non_retryable_http_{status}") from exc
            last_error = exc
        except (ValueError, KeyError) as exc:
            raise OllamaGenerationError("invalid_ollama_json") from exc
        if attempt + 1 < cfg["attempts"]:
            time.sleep(min(2 ** attempt, 5))
    raise OllamaGenerationError(
        f"ollama_unavailable_after_{cfg['attempts']}_attempts:"
        f"{type(last_error).__name__}"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "settings"])
    args = parser.parse_args()
    cfg = generation_settings()
    print(json.dumps(check_ollama(cfg) if args.command == "check" else cfg, indent=2))


if __name__ == "__main__":
    main()
