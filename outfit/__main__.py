"""Serve the outfit web app (styling advice, body measurements, sizes, try-on, avatars): python3 -m outfit [--host 0.0.0.0] [--port 8000]"""

import argparse
import base64
import binascii
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from . import avatar
from .analyze import OCCASIONS, analyze
from .body import measure
from .claude import AnalysisError
from .sizing import MEASURES, extract_chart, recommend
from .tryon import info as tryon_info
from .tryon import try_on

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


def data_url(data: bytes, media_type: str) -> str:
    return f"data:{media_type};base64,{base64.b64encode(data).decode()}"


class Handler(BaseHTTPRequestHandler):
    analyzer = staticmethod(analyze)
    chart_reader = staticmethod(extract_chart)
    dresser = staticmethod(try_on)
    avatar_maker = staticmethod(avatar.create_avatar)
    avatar_dresser = staticmethod(avatar.dress_avatar)

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
        elif self.path.startswith("/static/") and self.path.endswith(".js"):
            file = STATIC / self.path.removeprefix("/static/")
            if file.parent != STATIC or not file.is_file():
                return self._json(404, {"error": "找不到頁面"})
            self._send(200, file.read_bytes(), "text/javascript; charset=utf-8")
        elif self.path == "/api/occasions":
            self._json(200, OCCASIONS)
        elif self.path == "/api/avatar":
            saved = avatar.load_avatar()
            mine = saved and {"image": data_url(saved["image"], saved["media_type"]), "style": saved["style"]}
            self._json(200, {"styles": {k: label for k, (label, _) in avatar.STYLES.items()}, "avatar": mine})
        elif self.path == "/api/tryon-info":
            self._json(200, tryon_info())
        elif self.path == "/api/measures":
            self._json(200, {cat: {k: {"label": m.label, "hint": m.hint} for k, m in ms.items()} for cat, ms in MEASURES.items()})
        else:
            self._json(404, {"error": "找不到頁面"})

    def _analyze(self, body: dict) -> dict:
        image, media_type = decode_data_url(body.get("image", ""))
        return self.analyzer(image, media_type, body.get("occasion", "daily"), body.get("note", ""))

    def _size_chart(self, body: dict) -> dict:
        return self.chart_reader(*decode_data_url(body.get("image", "")))

    def _recommend_size(self, body: dict) -> dict:
        return recommend(body.get("chart") or {}, body.get("reference") or {})

    def _body_measure(self, body: dict) -> dict:
        return measure(body.get("front"), body.get("side"), body.get("height_cm"), body.get("calibration"))

    def _try_on(self, body: dict) -> dict:
        person, media_type = decode_data_url(body.get("image", ""))
        garments = [decode_data_url(g) for g in body.get("garments") or []]
        result = self.dresser(person, media_type, garments, body.get("description", ""), body.get("aspect"))
        return {"image": data_url(result["image"], result["media_type"]), "note": result["note"]}

    def _avatar_preview(self, body: dict) -> dict:
        result = self.avatar_maker(*decode_data_url(body.get("image", "")), body.get("style", ""))
        return {"image": data_url(result["image"], result["media_type"])}

    def _avatar_save(self, body: dict) -> dict:
        style = body.get("style", "")
        if style not in avatar.STYLES:
            raise AnalysisError(f"未知的風格：{style}")
        avatar.save_avatar(*decode_data_url(body.get("image", "")), style)
        return {"saved": True}

    def _avatar_dress(self, body: dict) -> dict:
        saved = avatar.load_avatar()
        if saved is None:
            raise AnalysisError("還沒有分身，請先產生並選一個分身。")
        garments = [decode_data_url(g) for g in body.get("garments") or []]
        result = self.avatar_dresser(saved["image"], saved["media_type"], saved["style"], garments,
                                     body.get("description", ""))
        return {"image": data_url(result["image"], result["media_type"])}

    ROUTES = {"/api/analyze": _analyze, "/api/size-chart": _size_chart, "/api/recommend-size": _recommend_size,
              "/api/body-measure": _body_measure, "/api/try-on": _try_on, "/api/avatar/preview": _avatar_preview,
              "/api/avatar/save": _avatar_save, "/api/avatar/dress": _avatar_dress}

    def do_POST(self):
        route = self.ROUTES.get(self.path)
        if route is None:
            return self._json(404, {"error": "找不到頁面"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "照片太大，請換一張小一點的"})
        try:
            body = json.loads(self.rfile.read(length))
            result = route(self, body)
        except (json.JSONDecodeError, AttributeError, TypeError):
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
