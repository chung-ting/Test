"""Virtual try-on: ask a Gemini image model to dress the person in a photo in other clothes.

The result is an illustration of the look, not a fit check: the model doesn't know the
garment's real size. Use the size recommendation for that.
"""

import logging
import os

from google import genai
from google.genai import errors, types

from .claude import MEDIA_TYPES, AnalysisError

MODEL = os.environ.get("OUTFIT_TRYON_MODEL", "gemini-3.1-flash-image")
MAX_GARMENTS = 3
log = logging.getLogger("outfit")

PROMPT = """Edit the first image, a photo of a real person, so that they are wearing the outfit described below.

Keep everything except the clothes exactly as in the original photo: the person's face and identity, hair,
skin tone, body shape and proportions, pose, hands, background, lighting and camera angle.
Replace only the clothing (and shoes or accessories if the outfit names them). Make the fabric, folds,
shadows and fit look like a real photograph. Do not make the person slimmer, taller or otherwise different.
Return one edited photo."""

GARMENT_NOTE = """The other {n} image(s) are product photos of the garments to put on the person.
Reproduce their color, pattern, material, logos and details faithfully; ignore any model or mannequin wearing them."""


def build_contents(person: bytes, media_type: str, garments: list[tuple[bytes, str]], description: str) -> list:
    if media_type not in MEDIA_TYPES or any(mt not in MEDIA_TYPES for _, mt in garments):
        raise AnalysisError("不支援的圖片格式")
    description = description.strip()
    if not description and not garments:
        raise AnalysisError("請描述想穿的衣服，或上傳衣服的商品照。")
    if len(garments) > MAX_GARMENTS:
        raise AnalysisError(f"衣服照片最多 {MAX_GARMENTS} 張。")
    prompt = PROMPT
    if garments:
        prompt += "\n\n" + GARMENT_NOTE.format(n=len(garments))
    prompt += "\n\nOutfit: " + (description or "the garments in the product photos")
    return [
        prompt,
        types.Part.from_bytes(data=person, mime_type=media_type),
        *(types.Part.from_bytes(data=data, mime_type=mt) for data, mt in garments),
    ]


def parse_response(response) -> dict:
    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise AnalysisError("這張照片或描述被 Gemini 的安全規則擋下了，請換一張照片或改一下描述。")
    for candidate in response.candidates or []:
        parts = candidate.content.parts if candidate.content else None
        texts = []
        for part in parts or []:
            if part.inline_data and part.inline_data.data:
                return {"image": part.inline_data.data, "media_type": part.inline_data.mime_type or "image/png",
                        "note": " ".join(texts)}
            if part.text:
                texts.append(part.text)
    raise AnalysisError("Gemini 沒有產生圖片，可能是安全規則擋下了。請換一張照片或改一下描述再試。")


def try_on(person: bytes, media_type: str, garments: list[tuple[bytes, str]], description: str,
           client=None) -> dict:
    contents = build_contents(person, media_type, garments, description)
    try:
        client = client or genai.Client()
    except ValueError:
        raise AnalysisError("伺服器還沒設定 GEMINI_API_KEY，無法產生試穿照。") from None
    try:
        response = client.models.generate_content(
            model=MODEL, contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
    except errors.APIError as e:
        log.exception("Gemini try-on call failed")
        if e.code in (401, 403) or (e.code == 400 and "API key" in str(e)):
            raise AnalysisError("Gemini API 金鑰無效或沒有權限，請確認 GEMINI_API_KEY。") from None
        if e.code == 404:
            raise AnalysisError(f"找不到模型 {MODEL}，請用 OUTFIT_TRYON_MODEL 設定可用的 Gemini 圖片模型。") from None
        if e.code == 429:
            raise AnalysisError("Gemini 使用量已達上限，請稍後再試，或確認帳號已開啟付費。") from None
        raise AnalysisError("Gemini 暫時無法使用，請稍後再試。") from None
    return parse_response(response)
