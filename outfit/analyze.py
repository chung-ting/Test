"""Send an outfit photo to Claude and get back a structured style analysis."""

import base64
import json

import anthropic

MODEL = "claude-opus-5-5"
MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
OCCASIONS = {"daily": "日常休閒", "work": "上班通勤", "date": "約會", "party": "聚會派對", "formal": "正式場合", "sport": "運動戶外"}

SYSTEM = """你是一位專業、友善的穿搭顧問。使用者會上傳一張自己的照片，請根據照片分析穿搭並給出建議。

原則：
- 全部用繁體中文回答，語氣正面、具體、實用。
- 談身形時只描述比例與線條（例如肩寬、腰線、腿長比例），用來說明哪些剪裁適合；不要評論胖瘦、外貌或給任何貶低的評語。
- 顏色建議依照片中可見的膚色、髮色判斷，並附上代表色的 hex 色碼。
- 建議要符合使用者指定的場合，單品要具體（例如「高腰直筒深藍牛仔褲」而不是「褲子」）。
- 如果照片裡看不到人或看不清楚穿著，把 is_person 設為 false，並在 summary 說明需要什麼樣的照片，其他欄位給空值即可。"""

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}


def _obj(**props):
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


SCHEMA = _obj(
    is_person={"type": "boolean"},
    summary=_STR,
    current_outfit=_obj(description=_STR, score={"type": "integer"}, strengths=_STR_LIST, improvements=_STR_LIST),
    body_and_color=_obj(
        body_notes=_STR,
        flattering_colors={"type": "array", "items": _obj(name=_STR, hex=_STR)},
        colors_to_avoid={"type": "array", "items": _obj(name=_STR, hex=_STR)},
    ),
    suggestions={
        "type": "array",
        "items": _obj(
            title=_STR,
            items={"type": "array", "items": _obj(category=_STR, item=_STR, color=_STR)},
            why=_STR,
        ),
    },
)


class AnalysisError(Exception):
    """Raised when the photo can't be analyzed; the message is safe to show users."""


def build_request(image: bytes, media_type: str, occasion: str, note: str = "") -> dict:
    if media_type not in MEDIA_TYPES:
        raise AnalysisError(f"不支援的圖片格式：{media_type}")
    if occasion not in OCCASIONS:
        raise AnalysisError(f"未知的場合：{occasion}")
    prompt = (
        f"場合：{OCCASIONS[occasion]}\n"
        + (f"補充說明：{note.strip()}\n" if note.strip() else "")
        + "請分析照片中的穿搭：給目前穿搭 1–10 分、優點與可改進處，說明適合的剪裁與顏色，"
        "再提出 3 套適合這個場合的完整穿搭建議。"
    )
    return {
        "model": MODEL,
        "max_tokens": 16000,
        "system": SYSTEM,
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
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


def parse_response(response) -> dict:
    if response.stop_reason == "refusal":
        raise AnalysisError("這張照片無法分析，請換一張試試。")
    if response.stop_reason == "max_tokens":
        raise AnalysisError("分析結果太長被截斷，請再試一次。")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        raise AnalysisError("沒有收到分析結果，請再試一次。")
    result = json.loads(text)
    score = result["current_outfit"]["score"]
    result["current_outfit"]["score"] = max(1, min(10, score))
    return result


def analyze(image: bytes, media_type: str, occasion: str, note: str = "", client=None) -> dict:
    client = client or anthropic.Anthropic()
    response = client.beta.messages.create(**build_request(image, media_type, occasion, note))
    return parse_response(response)
