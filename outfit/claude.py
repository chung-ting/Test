"""Shared helpers for sending a photo to Claude and getting schema-checked JSON back."""

import base64
import json

import anthropic

MODEL = "claude-opus-5-5"
MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class AnalysisError(Exception):
    """Raised when a photo can't be processed; the message is safe to show users."""


def obj(**props) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def build_request(system: str, image: bytes, media_type: str, prompt: str, schema: dict) -> dict:
    if media_type not in MEDIA_TYPES:
        raise AnalysisError(f"不支援的圖片格式：{media_type}")
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "system": system,
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
        "betas": ["server-side-fallback-2026-07-01"],
        "fallbacks": "default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.standard_b64encode(image).decode()}},
                {"type": "text", "text": prompt},
            ],
        }],
    }


def parse_json(response) -> dict:
    if response.stop_reason == "refusal":
        raise AnalysisError("這張照片無法分析，請換一張試試。")
    if response.stop_reason == "max_tokens":
        raise AnalysisError("分析結果太長被截斷，請再試一次。")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise AnalysisError("沒有收到分析結果，請再試一次。")
    return json.loads(text)


def call(request: dict, client=None) -> dict:
    client = client or anthropic.Anthropic()
    return parse_json(client.beta.messages.create(**request))
