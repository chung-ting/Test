"""Serve the outfit analysis web app: python3 -m outfit [--host 0.0.0.0] [--port 8000]"""

import argparse
import base64
import binascii
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from .analyze import OCCASIONS, AnalysisError, analyze

STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 15 * 1024 * 1024
log = logging.getLogger("outfit")


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    """Split a `data:image/jpeg;base64,...` URL into raw bytes and its media type."""
    header, sep, payload = data_url.partition(",")
    if not sep or not header.startswith("data:") or not header.endswith(";base64"):
        raise AnalysisError("圖片格式錯誤")
    try:
        return base64.b64decode(payload, validate=True), header[5:-7]
    except binascii.Error:
        raise AnalysisError("圖片格式錯誤") from None


class Handler(BaseHTTPRequestHandler):
    analyzer = staticmethod(analyze)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/occasions":
            self._json(200, OCCASIONS)
        else:
            self._json(404, {"error": "找不到頁面"})

    def do_POST(self):
        if self.path != "/api/analyze":
            return self._json(404, {"error": "找不到頁面"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "照片太大，請換一張小一點的"})
        try:
            body = json.loads(self.rfile.read(length))
            image, media_type = decode_data_url(body.get("image", ""))
            result = self.analyzer(image, media_type, body.get("occasion", "daily"), body.get("note", ""))
        except (json.JSONDecodeError, AttributeError):
            return self._json(400, {"error": "請求格式錯誤"})
        except AnalysisError as e:
            return self._json(400, {"error": str(e)})
        except anthropic.AuthenticationError:
            log.exception("Claude API authentication failed")
            return self._json(500, {"error": "伺服器的 API 金鑰設定有誤"})
        except anthropic.RateLimitError:
            return self._json(429, {"error": "使用的人太多了，請稍後再試"})
        except anthropic.APIError:
            log.exception("Claude API call failed")
            return self._json(502, {"error": "AI 服務暫時無法使用，請稍後再試"})
        self._json(200, result)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Outfit analysis web app")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow phones on the same Wi-Fi")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
