"""Avatars: turn a photo into a 3D-style character, try it in ready-made looks, and dress it.

The avatar is a picture rendered to look 3D, not a 3D model. Every outfit change redraws it with
the saved avatar as the reference, in the same fixed pose, so it stays recognisable.

The default 2D style is returned as a transparent PNG (the model paints a flat key color, rembg's
isnet-anime model cuts the character out). An optional style reference image steers its look.

Images come from Qwen-Image-Edit on its free Hugging Face Space when HF_TOKEN is set (shared,
queued, daily quota), otherwise from Gemini; OUTFIT_AVATAR_PROVIDER=qwen|flux|gemini overrides.
FLUX.1 Kontext takes a single image, so with "flux" dressing an avatar in garment photos uses Gemini.

Saved on this computer under OUTFIT_DATA_DIR (default ~/.outfit).

Command line:  python3 -m outfit.avatar photo.jpg [-o out_dir] [--style game] [--template city]
"""

import argparse
import io
import json
import logging
import mimetypes
import os
import tempfile
import time
from pathlib import Path

from google.genai import types

from .claude import MEDIA_TYPES, AnalysisError
from .tryon import generate_image

log = logging.getLogger("outfit")

QWEN_SPACE = "Qwen/Qwen-Image-Edit-2511"
FLUX_SPACE = "black-forest-labs/FLUX.1-Kontext-Dev"
QWEN_SIZE = (832, 1248)  # width, height: 2:3, multiples of 16
QWEN_MAX_IMAGES = 3      # the model is trained on one to three input images

# Styles chosen for broad appeal with women worldwide, modelled on what dress-up and life-sim games,
# avatar apps, animated films and collectible figures look like. "film2d" is the default: it keeps the
# person's look best, shows clothes with real proportions, and comes out on a transparent background.
STYLES = {
    "film2d": ("2D 動畫電影風", "a hand-drawn 2D animated character in a classic 2D animated-film aesthetic: clean "
               "contour lines, expressive eyes, simplified facial features, flat colors and subtle cel shading, "
               "with no 3D rendering, photorealistic textures or chibi proportions"),
    "game": ("3D 遊戲風", "a polished stylized 3D game character render, like the character models of a modern "
             "cozy life-simulation or open-world game (not a photograph): smooth slightly simplified CG skin, "
             "softly sculpted faces with clean shapes, gently enlarged clear eyes, glossy stylized hair clumps, "
             "clean physically based fabric materials, soft studio lighting with a rim light. Realistic adult "
             "proportions: the head about one seventh of the body height, long legs, natural shoulders, no big "
             "head, no chibi, no caricature"),
    "figure": ("療癒 Q 版公仔", "a kawaii chibi 3D collectible figurine: oversized head, small rounded body, rosy "
               "cheeks, gentle smile, soft pastel colors, matte vinyl-toy finish, soft studio lighting"),
    "fashion": ("時尚換裝風", "a polished 3D fashion dress-up game character (not a photograph): graceful "
                "proportions, large bright expressive eyes, soft glossy hair, luminous skin, gentle pastel studio "
                "lighting, and highly detailed fabric so the outfit is the star"),
}
# Ready-made looks to try right after taking a photo, drawn in the "film2d" style on a transparent
# background, so only the outfit changes (the scene is kept for the page to show behind the character).
# Each outfit names options for different people, and the model picks what suits whoever is in the photo.
TEMPLATES = {
    "city": ("城市街拍", "🏙️", "a trendy street-style outfit: an oversized blazer or bomber jacket over a fitted top, "
             "straight-leg jeans and clean white sneakers, a small crossbody bag",
             "a sunny city street with cafés, softly blurred"),
    "date": ("浪漫約會", "🌹", "an elegant date-night look: a satin midi slip dress with delicate heels, or a "
             "well-cut shirt with tailored trousers and loafers",
             "an evening rooftop terrace with warm string lights and a city skyline"),
    "office": ("質感通勤", "💼", "a polished office look: a tailored blazer, silk blouse or crisp shirt, wide-leg "
               "trousers and loafers, a structured tote bag",
               "a bright modern office lobby with plants"),
    "resort": ("海島度假", "🏝️", "a relaxed vacation look: flowy linen pieces in soft colors, a straw hat, "
               "sunglasses on the head and flat sandals",
               "a sunny beach boardwalk with palm trees and turquoise sea"),
    "campus": ("韓系校園", "📚", "a preppy campus look: a knit cardigan or varsity jacket, a collared shirt, a "
               "pleated skirt or chinos, loafers with white socks, a canvas tote",
               "an autumn university campus with golden ginkgo trees"),
    "winter": ("冬日聖誕", "❄️", "a cozy winter look: a chunky knit sweater, a long wool coat, a soft scarf and "
               "ankle boots",
               "a snowy street at dusk decorated with Christmas lights"),
}
# Every style keeps the person's own look; flattering, never "corrected".
LIKENESS = ("Keep every person clearly recognisable and flattering, but the same person: their real face shape, "
            "hairstyle and hair color, glasses or facial hair if any, skin tone (do not lighten or darken it) and "
            "body shape and build (do not make them thinner, taller or curvier). Tasteful and age-appropriate.")
FULL_BODY = ("Show everyone completely, from the top of the head down to the shoes, centered, standing upright "
             "facing the camera with arms relaxed, with a little empty space above the head and below the feet; "
             "extend any body parts the photo leaves out naturally. No text, logos, frames or packaging.")
ASPECT = 2 / 3
# Styles delivered as transparent PNGs: the model paints a flat key color (it can't output alpha),
# then a background-removal model trained on illustrations cuts the character out.
TRANSPARENT_STYLES = {"film2d"}
KEY_COLOR = (158, 230, 184)  # #9EE6B8, far from skin, hair and most garment colors
MAX_GARMENTS = 4


def data_dir() -> Path:
    return Path(os.environ.get("OUTFIT_DATA_DIR") or Path.home() / ".outfit")


def provider() -> str:
    chosen = os.environ.get("OUTFIT_AVATAR_PROVIDER", "").lower()
    if chosen in ("qwen", "flux", "gemini"):
        return chosen
    return "qwen" if os.environ.get("HF_TOKEN") else "gemini"


def dress_limit() -> int:
    """How many garment photos one outfit change can take with the current provider."""
    return QWEN_MAX_IMAGES - 1 if provider() == "qwen" else MAX_GARMENTS


def frame_for_full_body(photo: bytes, media_type: str) -> tuple[bytes, str]:
    """Put a close-up or half-body photo on a taller canvas so the model has room to draw the whole body.

    Image editors keep the input's framing, so a selfie stays a selfie unless the canvas grows.
    Tall (full-body) photos and anything that isn't a readable image are returned unchanged.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(photo)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return photo, media_type
    if img.width / img.height < 0.7:
        return photo, media_type
    W, H = 896, 1344
    canvas = Image.new("RGB", (W, H), (246, 228, 234))
    w = int(W * 0.62)
    h = min(int(img.height * w / img.width), int(H * 0.5))
    w = int(img.width * h / img.height)
    canvas.paste(img.resize((w, h)), ((W - w) // 2, int(H * 0.03)))
    out = io.BytesIO()
    canvas.save(out, "JPEG", quality=92)
    return out.getvalue(), "image/jpeg"


# Prompts here avoid naming example features or accessories ("such as glasses"): image models tend
# to draw whatever a prompt names, even when it says to keep or avoid it.
def film2d_prompt(outfit: str | None = None, has_reference: bool = False) -> str:
    """The 2D animated-film prompt; Image 2, when given, is only a style reference."""
    reference = ", using Image 2 only as a reference for the illustration style" if has_reference else ""
    if outfit:
        clothing = (f"Dress the person in {outfit}, choosing what suits them, and keep any accessories they "
                    "already wear. Preserve the original pose and expression.")
    else:
        clothing = ("Keep the original outfit and accessories unchanged, including garment shapes, colors, "
                    "patterns, fit and any accessories. Preserve the original pose and expression.")
    if has_reference:
        clothing += " Do not copy the outfit or identity from Image 2."
    return (f"Transform the person in Image 1 into a full-body, hand-drawn 2D animated character{reference}.\n\n"
            "STYLE:\nClassic 2D animated-film aesthetic, clean contour lines, expressive eyes, simplified facial "
            "features, flat colors, and subtle cel shading. No 3D rendering, photorealistic textures, or chibi "
            "proportions.\n\n"
            "IDENTITY:\nPreserve the person's recognizable facial features, face shape, skin tone, hairstyle, hair "
            "color, smile, and distinctive details. Keep their apparent age and natural body proportions. Do not "
            "replace their face with a generic cartoon face. Copy the hair color and the skin color from Image 1 "
            "exactly, at the same lightness; black hair stays black. Draw only features and accessories that are "
            "visible in Image 1.\n\n"
            f"CLOTHING AND POSE:\n{clothing}\n\n"
            "FRAMING:\nShow the entire person from head to toe, with space around the hair, hands, accessories, "
            "and feet. If the original photo crops the lower legs, naturally extend the existing clothing and add "
            "simple shoes that match the outfit.\n\n"
            "BACKGROUND:\nA completely flat, solid mint-green background (#9EE6B8) with nothing else in it: no "
            "shadow, glow, halo, text, watermark, or border.")


def style_prompt(style: str, has_reference: bool = False) -> str:
    if style == "film2d":
        return film2d_prompt(None, has_reference)
    return (f"Redraw the person or people in this picture as {STYLES[style][1]}. {LIKENESS} Keep the clothes they "
            f"are wearing, with the same colors. {FULL_BODY} Plain soft pastel background.")


def template_prompt(template: str, has_reference: bool = False) -> str:
    return film2d_prompt(TEMPLATES[template][2], has_reference)


def remove_background(result: dict) -> dict:
    """Cut the character out of a flat key-color background; returns a transparent PNG.

    Leaves the result unchanged when it isn't a readable image or rembg isn't installed.
    """
    try:
        from PIL import Image, UnidentifiedImageError
        from rembg import new_session, remove
    except ImportError:
        log.warning("rembg is not installed; keeping the background")
        return result
    try:
        img = Image.open(io.BytesIO(result["image"])).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return result
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session("isnet-anime")
    cut = trim_to_character(remove(img, session=_rembg_session, post_process_mask=True))
    out = io.BytesIO()
    cut.save(out, "PNG")
    return {**result, "image": out.getvalue(), "media_type": "image/png"}


def trim_to_character(cut, margin: float = 0.04):
    """Crop a cut-out to the character plus a small margin, so it fills the frame it is shown in."""
    box = cut.getchannel("A").getbbox()
    if box is None:
        return cut
    pad = int(max(box[2] - box[0], box[3] - box[1]) * margin)
    return cut.crop((max(box[0] - pad, 0), max(box[1] - pad, 0),
                     min(box[2] + pad, cut.width), min(box[3] + pad, cut.height)))


_rembg_session = None


def on_key_color(image: bytes, media_type: str) -> tuple[bytes, str]:
    """Flatten a transparent avatar onto the key color, so the model sees the background it should keep."""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(image))
    except (UnidentifiedImageError, OSError):
        return image, media_type
    if img.mode not in ("RGBA", "LA", "P"):
        return image, media_type
    img = img.convert("RGBA")
    flat = Image.new("RGB", img.size, KEY_COLOR)
    flat.paste(img, (0, 0), img)
    out = io.BytesIO()
    flat.save(out, "PNG")
    return out.getvalue(), "image/png"


def _space(space: str, images: list[tuple[bytes, str]], prompt: str) -> dict:
    """Run an image-editing demo Space on Hugging Face; both take image file(s) plus a prompt."""
    from gradio_client import Client, handle_file
    from gradio_client.exceptions import AppError

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (data, media_type) in enumerate(images):
            path = Path(tmp) / f"input{i}{mimetypes.guess_extension(media_type) or '.jpg'}"
            path.write_bytes(data)
            paths.append(str(path))
        try:
            client = Client(space, token=os.environ.get("HF_TOKEN") or None, verbose=False)
            if space == QWEN_SPACE:
                gallery = [{"image": handle_file(p), "caption": None} for p in paths]
                # rewrite_prompt=False: our prompts already spell out what to keep.
                result, _seed = client.predict(gallery, prompt, 0, True, 4.0, 30, QWEN_SIZE[1], QWEN_SIZE[0], False,
                                               api_name="/infer")
                result = result[0]["image"]
            else:
                result, _seed = client.predict(handle_file(paths[0]), prompt, 0, True, 2.5, 28, api_name="/infer")
        except AppError as e:
            log.warning("%s refused: %s", space, e)
            if "quota" in str(e).lower() or "limit" in str(e).lower():
                raise AnalysisError("今天的免費產圖次數用完了，請明天再試，或改用付費服務。") from None
            raise AnalysisError("免費產圖服務這次沒有產生圖片，請換一張照片或稍後再試。") from None
        except Exception:
            log.exception("%s call failed", space)
            raise AnalysisError("免費產圖服務暫時連不上，請稍後再試。") from None
    path = Path(result["path"] if isinstance(result, dict) else result)
    return {"image": path.read_bytes(), "media_type": mimetypes.guess_type(path.name)[0] or "image/webp", "note": ""}


def _generate(images: list[tuple[bytes, str]], prompt: str, client=None) -> dict:
    if provider() == "qwen":
        return _space(QWEN_SPACE, images[:QWEN_MAX_IMAGES], prompt)
    if provider() == "flux":
        return _space(FLUX_SPACE, images[:1], prompt)  # FLUX.1 Kontext reads a single image
    parts = [prompt, *(types.Part.from_bytes(data=d, mime_type=mt) for d, mt in images)]
    return generate_image(parts, ASPECT, client)


def render(photo: bytes, media_type: str, prompt: str, client=None,
           style_ref: tuple[bytes, str] | None = None, transparent: bool = False) -> dict:
    if media_type not in MEDIA_TYPES or (style_ref and style_ref[1] not in MEDIA_TYPES):
        raise AnalysisError("不支援的圖片格式")
    images = [frame_for_full_body(photo, media_type), *([style_ref] if style_ref else [])]
    result = _generate(images, prompt, client)
    return remove_background(result) if transparent else result


def _use_reference(style_ref) -> bool:
    return bool(style_ref) and provider() != "flux"


def create_avatar(photo: bytes, media_type: str, style: str, client=None,
                  style_ref: tuple[bytes, str] | None = None) -> dict:
    if style not in STYLES:
        raise AnalysisError(f"未知的風格：{style}")
    has_ref = style == "film2d" and _use_reference(style_ref)
    return render(photo, media_type, style_prompt(style, has_ref), client, style_ref if has_ref else None,
                  transparent=style in TRANSPARENT_STYLES)


def try_template(photo: bytes, media_type: str, template: str, client=None,
                 style_ref: tuple[bytes, str] | None = None) -> dict:
    if template not in TEMPLATES:
        raise AnalysisError(f"未知的模板：{template}")
    has_ref = _use_reference(style_ref)
    return render(photo, media_type, template_prompt(template, has_ref), client, style_ref if has_ref else None,
                  transparent=True)


def dress_avatar(avatar: bytes, avatar_type: str, style: str, garments: list[tuple[bytes, str]],
                 description: str, client=None) -> dict:
    description = description.strip()
    if not garments and not description:
        raise AnalysisError("請上傳衣服照片，或描述想穿什麼。")
    limit = dress_limit()
    if len(garments) > limit:
        raise AnalysisError(f"一次最多 {limit} 件。")
    if any(mt not in MEDIA_TYPES for _, mt in [(avatar, avatar_type), *garments]):
        raise AnalysisError("不支援的圖片格式")
    transparent = style in TRANSPARENT_STYLES
    if transparent:
        avatar, avatar_type = on_key_color(avatar, avatar_type)
    prompt = (f"The first image is a character drawn as {STYLES.get(style, STYLES['film2d'])[1]}. Redraw exactly "
              "the same character, same face, hair, body, art style, pose, framing and background, but wearing "
              "the outfit below. Translate each garment into the character's art style while keeping its "
              "colors, patterns and details. Replace only the items given; keep the rest of the current outfit. "
              f"{FULL_BODY}")
    if garments:
        prompt += f"\n\nThe other {len(garments)} image(s) are photos of the garments, one item each."
    if description:
        prompt += f"\n\nOutfit: {description}"
    if transparent:
        prompt += "\n\nKeep the completely flat, solid mint-green background (#9EE6B8)."
    result = _generate([(avatar, avatar_type), *garments], prompt, client)
    return remove_background(result) if transparent else result


def save_avatar(image: bytes, media_type: str, style: str) -> None:
    folder = data_dir()
    folder.mkdir(parents=True, exist_ok=True)
    ext = mimetypes.guess_extension(media_type) or ".png"
    for old in folder.glob("avatar.*"):
        old.unlink()
    (folder / f"avatar{ext}").write_bytes(image)
    (folder / "avatar.json").write_text(json.dumps({"style": style, "media_type": media_type, "file": f"avatar{ext}",
                                                    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}))


def load_avatar() -> dict | None:
    meta_file = data_dir() / "avatar.json"
    if not meta_file.is_file():
        return None
    meta = json.loads(meta_file.read_text())
    return {**meta, "image": (data_dir() / meta["file"]).read_bytes()}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Render a photo as avatars in each style, or in a template look")
    parser.add_argument("photo", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("."))
    parser.add_argument("--style", choices=[*STYLES, "all"])
    parser.add_argument("--template", choices=[*TEMPLATES, "all"])
    args = parser.parse_args(argv)
    media_type = mimetypes.guess_type(args.photo.name)[0] or "image/jpeg"
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.template:
        names = TEMPLATES if args.template == "all" else [args.template]
        jobs += [(f"template-{t}", TEMPLATES[t][0], lambda p, t=t: try_template(p, media_type, t)) for t in names]
    if args.style or not args.template:
        names = STYLES if args.style in (None, "all") else [args.style]
        jobs += [(s, STYLES[s][0], lambda p, s=s: create_avatar(p, media_type, s)) for s in names]
    photo = args.photo.read_bytes()
    for name, label, run in jobs:
        try:
            result = run(photo)
        except AnalysisError as e:
            print(f"{name}: {e}")
            continue
        out = args.out / f"{args.photo.stem}-{name}{mimetypes.guess_extension(result['media_type']) or '.png'}"
        out.write_bytes(result["image"])
        print(f"{name} ({label}): {out}", flush=True)


if __name__ == "__main__":
    main()
