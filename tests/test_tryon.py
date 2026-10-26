import base64
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from google import genai
from google.genai import types

from outfit import AnalysisError
from outfit.tryon import MODEL, build_contents, try_on

PERSON = b"\xff\xd8person-jpeg"
SHIRT = b"\x89PNGshirt"
RESULT = b"\x89PNGdressed"


class FakeGemini(BaseHTTPRequestHandler):
    """Stands in for the Gemini REST API so the real SDK runs end to end."""

    requests = []
    status = 200
    reply = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append((self.path, body))
        payload = json.dumps(type(self).reply).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def image_reply(data=RESULT):
    return {"candidates": [{"content": {"role": "model", "parts": [
        {"text": "Here is the edited photo."},
        {"inlineData": {"mimeType": "image/png", "data": base64.b64encode(data).decode()}},
    ]}, "finishReason": "STOP"}]}


class TryOnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGemini)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeGemini.requests, FakeGemini.status, FakeGemini.reply = [], 200, image_reply()
        self.client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
            base_url=f"http://127.0.0.1:{self.server.server_address[1]}", retry_options=types.HttpRetryOptions(attempts=1)))

    def test_sends_photos_and_returns_image(self):
        result = try_on(PERSON, "image/jpeg", [(SHIRT, "image/png")], "白色牛津襯衫", client=self.client)
        self.assertEqual(result, {"image": RESULT, "media_type": "image/png", "note": "Here is the edited photo."})
        path, body = FakeGemini.requests[0]
        self.assertIn(f"models/{MODEL}:generateContent", path)
        parts = body["contents"][0]["parts"]
        self.assertIn("白色牛津襯衫", parts[0]["text"])
        self.assertIn("product photos", parts[0]["text"])
        self.assertEqual([base64.urlsafe_b64decode(p["inlineData"]["data"] + "==") for p in parts[1:]], [PERSON, SHIRT])
        self.assertEqual(body["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"])

    def test_no_image_in_reply_is_explained(self):
        FakeGemini.reply = {"candidates": [{"content": {"role": "model", "parts": [{"text": "I can't do that."}]},
                                            "finishReason": "STOP"}]}
        with self.assertRaisesRegex(AnalysisError, "沒有產生圖片"):
            try_on(PERSON, "image/jpeg", [], "紅色洋裝", client=self.client)
        FakeGemini.reply = {"promptFeedback": {"blockReason": "SAFETY"}}
        with self.assertRaisesRegex(AnalysisError, "安全規則"):
            try_on(PERSON, "image/jpeg", [], "紅色洋裝", client=self.client)

    def test_api_errors_become_friendly_messages(self):
        cases = {404: "找不到模型", 403: "金鑰", 429: "上限", 500: "暫時無法使用"}
        for status, message in cases.items():
            FakeGemini.status = status
            FakeGemini.reply = {"error": {"code": status, "message": "boom", "status": "X"}}
            with self.subTest(status=status), self.assertRaisesRegex(AnalysisError, message):
                try_on(PERSON, "image/jpeg", [], "紅色洋裝", client=self.client)

    def test_validates_input_before_calling(self):
        with self.assertRaisesRegex(AnalysisError, "描述"):
            build_contents(PERSON, "image/jpeg", [], "  ")
        with self.assertRaisesRegex(AnalysisError, "最多"):
            build_contents(PERSON, "image/jpeg", [(SHIRT, "image/png")] * 4, "")
        with self.assertRaisesRegex(AnalysisError, "格式"):
            build_contents(PERSON, "image/bmp", [], "洋裝")
        self.assertEqual(FakeGemini.requests, [])


class TryOnRouteTests(unittest.TestCase):
    def test_route_decodes_photos_and_returns_data_url(self):
        import urllib.request
        from outfit.__main__ import Handler

        calls = []

        def fake(person, media_type, garments, description):
            calls.append((person, media_type, garments, description))
            return {"image": RESULT, "media_type": "image/png", "note": ""}

        handler = type("H", (Handler,), {"dresser": staticmethod(fake), "log_message": lambda *a: None})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = lambda data, mt: f"data:{mt};base64," + base64.b64encode(data).decode()
            body = {"image": url(PERSON, "image/jpeg"), "garments": [url(SHIRT, "image/png")], "description": "襯衫"}
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/try-on",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                reply = json.load(r)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(calls, [(PERSON, "image/jpeg", [(SHIRT, "image/png")], "襯衫")])
        self.assertEqual(reply["image"], url(RESULT, "image/png"))


if __name__ == "__main__":
    unittest.main()
