"""Virtual try-on through the Genlook API: https://docs.genlook.app/docs/tryon-api/introduction

Genlook is built for clothing try-on and needs a product photo of each garment; a text
description alone isn't enough. Set GENLOOK_MOCK=1 to use its free test mode, which skips the
AI and returns the person photo unchanged.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid

from .claude import MEDIA_TYPES, AnalysisError

BASE_URL = os.environ.get("GENLOOK_API_BASE", "https://api.genlook.app/tryon/v1")
MAX_PRODUCTS = 4
POLL_SECONDS = 90
log = logging.getLogger("outfit")

# What to tell the user for each Genlook error code (https://docs.genlook.app/docs/tryon-api/errors).
MESSAGES = {
    "MISSING_API_KEY": "伺服器還沒設定 GENLOOK_API_KEY，無法產生試穿照。",
    "INVALID_API_KEY": "Genlook API 金鑰無效，請確認 GENLOOK_API_KEY。",
    "INSUFFICIENT_CREDITS": "Genlook 點數用完了，請到 platform.genlook.app 儲值。",
    "QUOTA_EXCEEDED": "Genlook 使用量暫時達到上限，請稍後再試。",
    "RATE_LIMITED": "請求太頻繁，請稍後再試。",
    "CONTENT_POLICY_VIOLATION": "這件衣服不能用來試穿（泳裝、內衣類需要已付費的 Genlook 帳號）。",
    "MULTI_PRODUCT_NOT_SUPPORTED": "多件一起試穿時每件都必須是衣服；帽子、鞋子、眼鏡、飾品請單獨試穿。",
    "MISSING_REQUIRED_CLASSIFICATION": "這類衣服需要一張「有人穿著」的商品照，請再加一張。",
    "FILE_TOO_LARGE": "圖片太大（上限 10 MB），請換小一點的照片。",
    "UNSUPPORTED_IMAGE_TYPE": "圖片格式不支援，請用 JPEG、PNG 或 WebP。",
    "INVALID_IMAGE": "圖片無法讀取，請換一張。",
    "GENERATION_FAILED": "試穿照產生失敗（點數已退還），請再試一次。",
    "GENERATION_TIMEOUT": "試穿照產生逾時（點數已退還），請再試一次。",
    "INTERNAL_ERROR": "Genlook 暫時出錯，請稍後再試。",
}


class GenlookError(Exception):
    def __init__(self, code: str, message: str = "", status: int = 0):
        super().__init__(f"{code}: {message}")
        self.code = code


def _multipart(fields: dict[str, str], files: dict[str, tuple[bytes, str]]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    out = bytearray()
    for name, value in fields.items():
        out += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
    for name, (data, media_type) in files.items():
        ext = media_type.split("/")[-1]
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{name}.{ext}"\r\n'
                f"Content-Type: {media_type}\r\n\r\n").encode() + data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class Client:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, sleep=time.sleep):
        self.api_key = api_key if api_key is not None else os.environ.get("GENLOOK_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.sleep = sleep

    def request(self, method: str, path: str, *, fields=None, files=None, timeout: float = 30) -> tuple[int, dict]:
        headers = {"x-api-key": self.api_key}
        body = None
        if fields is not None or files is not None:
            body, headers["Content-Type"] = _multipart(fields or {}, files or {})
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            try:
                err = json.load(e)
            except ValueError:
                err = {}
            raise GenlookError(err.get("code", f"HTTP_{e.code}"), err.get("message", ""), e.code) from None

    def upload(self, image: bytes, media_type: str) -> str:
        # crop=false keeps the photo's framing, so the result lines up with it (see keepface.js).
        _, data = self.request("POST", "/images/upload", fields={"crop": "false", "keepForDays": "1"},
                               files={"file": (image, media_type)})
        return data["imageId"]

    def try_on(self, image_id: str, garments: list[tuple[bytes, str]], description: str, mock: bool) -> str:
        """Runs the try-on and returns the result image URL."""
        products = [{"title": f"Garment {i + 1}", "images": [{"source": {"fileKey": f"product{i}"}}],
                     **({"description": description} if description else {})} for i in range(len(garments))]
        payload = {"products": products, "person": {"image": {"source": {"id": image_id}}}, "mock": mock}
        files = {f"product{i}": g for i, g in enumerate(garments)}
        status, data = self.request("POST", "/try-on/sync", fields={"data": json.dumps(payload)}, files=files,
                                    timeout=100)
        deadline = time.monotonic() + POLL_SECONDS
        while status == 202 or data.get("status") in ("PENDING", "PROCESSING"):
            if time.monotonic() > deadline:
                raise GenlookError("GENERATION_TIMEOUT")
            self.sleep(1)
            status, data = self.request("GET", f"/generations/{data['generationId']}")
        if data.get("status") == "FAILED":
            raise GenlookError(data.get("errorCode", "GENERATION_FAILED"), data.get("errorMessage", ""))
        return data["resultImageUrl"]


def _download(url: str) -> tuple[bytes, str]:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read(), r.headers.get_content_type()


def try_on(person: bytes, media_type: str, garments: list[tuple[bytes, str]], description: str,
           aspect: float | None = None, client: Client | None = None, download=_download) -> dict:
    if not garments:
        raise AnalysisError("Genlook 需要衣服的商品照，請至少上傳一張（每張一件衣服）。")
    if len(garments) > MAX_PRODUCTS:
        raise AnalysisError(f"一次最多 {MAX_PRODUCTS} 件衣服。")
    if media_type not in MEDIA_TYPES or any(mt not in MEDIA_TYPES for _, mt in garments):
        raise AnalysisError("不支援的圖片格式")
    client = client or Client()
    if not client.api_key:
        raise AnalysisError(MESSAGES["MISSING_API_KEY"])
    mock = os.environ.get("GENLOOK_MOCK") == "1"
    try:
        url = client.try_on(client.upload(person, media_type), garments, description.strip(), mock)
        image, result_type = download(url)
    except GenlookError as e:
        log.warning("Genlook try-on failed: %s", e)
        raise AnalysisError(MESSAGES.get(e.code, f"Genlook 回傳錯誤：{e.code}")) from None
    except (urllib.error.URLError, TimeoutError):
        log.exception("Genlook request failed")
        raise AnalysisError("連不上 Genlook，請稍後再試。") from None
    return {"image": image, "media_type": result_type, "note": "（測試模式：照片未經 AI 處理）" if mock else ""}
