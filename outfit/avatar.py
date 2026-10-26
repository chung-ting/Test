"""Cute avatars: turn a photo into a 3D-style character, then dress it in other clothes.

The avatar is a picture rendered to look 3D, not a 3D model. Every outfit change redraws it with
the saved avatar as the reference, in the same fixed pose, so it stays recognisable.

Saved on this computer under OUTFIT_DATA_DIR (default ~/.outfit).

Command line, to try the styles on a photo:  python3 -m outfit.avatar photo.jpg [-o out_dir] [--style figure]
"""

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path

from google.genai import types

from .claude import MEDIA_TYPES, AnalysisError
from .tryon import generate_image

STYLES = {
    "figure": ("Q 版 3D 公仔", "a cute chibi 3D collectible vinyl figure: oversized head, small rounded body, "
               "glossy toy-like materials, soft studio lighting"),
    "animation": ("3D 動畫電影風", "a stylized 3D animated feature-film character: natural body proportions, "
                  "expressive face, soft cinematic lighting, detailed fabric"),
    "game": ("遊戲角色風", "a cheerful 3D life-simulation game character: simple rounded shapes, bright clean "
             "colors, smooth matte shading"),
}
POSE = ("Full body from head to shoes, standing upright facing the camera, arms relaxed slightly away from "
        "the body, on a plain light grey background. No text, logos, frames or packaging.")
ASPECT = 3 / 4
MAX_GARMENTS = 4


def data_dir() -> Path:
    return Path(os.environ.get("OUTFIT_DATA_DIR") or Path.home() / ".outfit")


def _image_part(data: bytes, media_type: str):
    if media_type not in MEDIA_TYPES:
        raise AnalysisError("不支援的圖片格式")
    return types.Part.from_bytes(data=data, mime_type=media_type)


def create_avatar(photo: bytes, media_type: str, style: str, client=None) -> dict:
    if style not in STYLES:
        raise AnalysisError(f"未知的風格：{style}")
    prompt = (f"Turn the person in this photo into {STYLES[style][1]}. Keep what makes them recognisable: "
              "face shape, hairstyle and hair color, skin tone, glasses or facial hair if any, and the clothes "
              f"they are wearing with the same colors. {POSE}")
    return generate_image([prompt, _image_part(photo, media_type)], ASPECT, client)


def dress_avatar(avatar: bytes, avatar_type: str, style: str, garments: list[tuple[bytes, str]],
                 description: str, client=None) -> dict:
    description = description.strip()
    if not garments and not description:
        raise AnalysisError("請上傳衣服照片，或描述想穿什麼。")
    if len(garments) > MAX_GARMENTS:
        raise AnalysisError(f"一次最多 {MAX_GARMENTS} 件。")
    prompt = (f"The first image is a character drawn as {STYLES.get(style, STYLES['figure'])[1]}. Redraw exactly "
              "the same character, same face, hair, body, art style, pose, framing and background, but wearing "
              "the outfit below. Translate each garment into the character's art style while keeping its "
              "colors, patterns and details. Replace only the items given; keep the rest of the current outfit. "
              f"{POSE}")
    if garments:
        prompt += f"\n\nThe other {len(garments)} image(s) are photos of the garments, one item each."
    if description:
        prompt += f"\n\nOutfit: {description}"
    parts = [prompt, _image_part(avatar, avatar_type), *(_image_part(d, mt) for d, mt in garments)]
    return generate_image(parts, ASPECT, client)


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
    parser = argparse.ArgumentParser(description="Render a photo as cute avatars in each style")
    parser.add_argument("photo", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=Path("."))
    parser.add_argument("--style", choices=[*STYLES, "all"], default="all")
    args = parser.parse_args(argv)
    media_type = mimetypes.guess_type(args.photo.name)[0] or "image/jpeg"
    args.out.mkdir(parents=True, exist_ok=True)
    for style in STYLES if args.style == "all" else [args.style]:
        try:
            result = create_avatar(args.photo.read_bytes(), media_type, style)
        except AnalysisError as e:
            print(f"{style}: {e}")
            continue
        out = args.out / f"{args.photo.stem}-{style}{mimetypes.guess_extension(result['media_type']) or '.png'}"
        out.write_bytes(result["image"])
        print(f"{style} ({STYLES[style][0]}): {out}")


if __name__ == "__main__":
    main()
