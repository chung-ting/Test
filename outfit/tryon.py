"""Virtual try-on: dress the person in a photo in other clothes, with Genlook or a Gemini image model.

OUTFIT_TRYON_PROVIDER picks the service ("genlook" or "gemini"); by default Genlook when
GENLOOK_API_KEY is set, otherwise Gemini. Either way the result is an illustration of the look,
not a fit check: the model doesn't know the garment's real size. Use the size recommendation for that.
"""

import logging
import math
import os

from google import genai
from google.genai import errors, types

from . import genlook
from .claude import MEDIA_TYPES, AnalysisError

MODEL = os.environ.get("OUTFIT_TRYON_MODEL", "gemini-3.1-flash-image")
MAX_GARMENTS = 3
# Aspect ratios the Gemini image models accept for their output.
ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
log = logging.getLogger("outfit")

PROMPT = """Edit the first image, a photo of a real person, so that they are wearing the outfit described below.

Keep everything except the clothes exactly as in the original photo: the person's face and identity, hair,
skin tone, body shape and proportions, pose, hands, background, lighting, framing and camera angle.
The face must stay pixel-for-pixel the same person: do not redraw, beautify, age, or restyle it.
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


def nearest_aspect_ratio(aspect: float) -> str:
    """The supported output ratio closest to width / height, so the edit keeps the photo's framing."""
    def value(ratio: str) -> float:
        w, h = ratio.split(":")
        return int(w) / int(h)
    return min(ASPECT_RATIOS, key=lambda r: abs(math.log(value(r) / aspect)))


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


def provider() -> str:
    chosen = os.environ.get("OUTFIT_TRYON_PROVIDER", "").lower()
    if chosen in ("genlook", "gemini"):
        return chosen
    return "genlook" if os.environ.get("GENLOOK_API_KEY") else "gemini"


def info() -> dict:
    """What the page needs to know about the active service."""
    if provider() == "genlook":
        return {"provider": "genlook", "needs_garment_photo": True, "max_garments": genlook.MAX_PRODUCTS}
    return {"provider": "gemini", "needs_garment_photo": False, "max_garments": MAX_GARMENTS}


def try_on(person: bytes, media_type: str, garments: list[tuple[bytes, str]], description: str,
           aspect: float | None = None) -> dict:
    if provider() == "genlook":
        return genlook.try_on(person, media_type, garments, description, aspect)
    return try_on_gemini(person, media_type, garments, description, aspect)


def try_on_gemini(person: bytes, media_type: str, garments: list[tuple[bytes, str]], description: str,
                  aspect: float | None = None, client=None) -> dict:
    return generate_image(build_contents(person, media_type, garments, description), aspect, client)


def generate_image(contents: list, aspect: float | None = None, client=None) -> dict:
    """Run a Gemini image model on the prompt and images; returns {image, media_type, note}."""
    image_config = None
    if isinstance(aspect, (int, float)) and not isinstance(aspect, bool) and 0.2 < aspect < 5:
        image_config = types.ImageConfig(aspect_ratio=nearest_aspect_ratio(aspect))
    try:
        client = client or genai.Client()
    except ValueError:
        raise AnalysisError("伺服器還沒設定 GEMINI_API_KEY，無法產生圖片。") from None
    try:
        response = client.models.generate_content(
            model=MODEL, contents=contents,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"], image_config=image_config),
        )
    except errors.APIError as e:
        log.exception("Gemini image call failed")
        if e.code in (401, 403) or (e.code == 400 and "API key" in str(e)):
            raise AnalysisError("Gemini API 金鑰無效或沒有權限，請確認 GEMINI_API_KEY。") from None
        if e.code == 404:
            raise AnalysisError(f"找不到模型 {MODEL}，請用 OUTFIT_TRYON_MODEL 設定可用的 Gemini 圖片模型。") from None
        if e.code == 429:
            raise AnalysisError("Gemini 使用量已達上限，請稍後再試，或確認帳號已開啟付費。") from None
        raise AnalysisError("Gemini 暫時無法使用，請稍後再試。") from None
    return parse_response(response)
