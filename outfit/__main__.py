"""Serve the outfit web app (styling advice, body measurements, sizes, try-on): python3 -m outfit [--host 0.0.0.0] [--port 8000]"""

import argparse
import base64
import binascii
import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anthropic

from .analyze import OCCASIONS, analyze
from .body import measure
from .claude import AnalysisError
from .sizing import MEASURES, extract_chart, recommend
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


class Handler(BaseHTTPRequestHandler):
    analyzer = staticmethod(analyze)
    chart_reader = staticmethod(extract_chart)
    dresser = staticmethod(try_on)

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
        result = self.dresser(person, media_type, garments, body.get("description", ""))
        image = base64.b64encode(result["image"]).decode()
        return {"image": f"data:{result['media_type']};base64,{image}", "note": result["note"]}

    ROUTES = {"/api/analyze": _analyze, "/api/size-chart": _size_chart, "/api/recommend-size": _recommend_size,
              "/api/body-measure": _body_measure, "/api/try-on": _try_on}

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
