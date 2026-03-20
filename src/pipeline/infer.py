"""
Low-level VLM call to the llama.cpp OpenAI-compatible server.

Sends a document image + system prompt and returns the raw text response.
JSON extraction / validation is handled by higher-level modules.
"""
from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent.parent / ".env")

_SERVER_URL  = os.getenv("MODEL_SERVER_URL", "http://localhost:8080/v1")
_API_KEY     = os.getenv("OPENAI_API_KEY",   "local-llama-key")
_MODEL_NAME  = os.getenv("MODEL_NAME",        "qwen2.5-vl-7b-q4_k_m")
_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
_MAX_TOKENS  = int(os.getenv("MAX_TOKENS",    "2048"))


def _client() -> OpenAI:
    return OpenAI(api_key=_API_KEY, base_url=_SERVER_URL)


def call_vlm(
    image_b64: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = _TEMPERATURE,
    max_tokens: int = _MAX_TOKENS,
) -> tuple[str, int]:
    """
    Send an image + prompts to the VLM server.

    Returns:
        (response_text, latency_ms)
    """
    client = _client()
    t0 = time.monotonic()

    response = client.chat.completions.create(
        model=_MODEL_NAME,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_b64},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            },
        ],
    )

    latency_ms = int((time.monotonic() - t0) * 1000)
    text = response.choices[0].message.content or ""
    return text, latency_ms


def parse_json_from_response(text: str) -> dict:
    """
    Extract JSON from model output even when wrapped in markdown code fences.
    """
    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        text = match.group(1)

    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find first { ... } block
        brace_match = re.search(r"\{[\s\S]+\}", text)
        if brace_match:
            return json.loads(brace_match.group())
        raise ValueError(f"Could not parse JSON from model response:\n{text[:500]}")
