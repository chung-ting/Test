import email
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from outfit import AnalysisError, genlook, tryon

PERSON = b"\xff\xd8person-jpeg"
SHIRT = b"\x89PNGshirt"
JEANS = b"\x89PNGjeans"
RESULT = b"\xff\xd8dressed-jpeg"


def parse_multipart(content_type: str, body: bytes) -> dict:
    msg = email.message_from_bytes(f"Content-Type: {content_type}\r\n\r\n".encode() + body)
    return {part.get_param("name", header="content-disposition"): (part.get_payload(decode=True), part.get_content_type())
            for part in msg.get_payload()}


class FakeGenlook(BaseHTTPRequestHandler):
    """Stands in for api.genlook.app, following the documented request and response shapes."""

    calls = []
    mode = "ok"
    polls = 0

    def reply(self, status, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        cls = type(self)
        if self.headers["x-api-key"] != "gk_test":
            return self.reply(401, {"code": "INVALID_API_KEY", "message": "bad key", "status": 401})
        parts = parse_multipart(self.headers["Content-Type"], self.rfile.read(int(self.headers["Content-Length"])))
        cls.calls.append((self.path, parts))
        if self.path == "/images/upload":
            return self.reply(201, {"imageId": "img-1", "imageUrl": "https://example.invalid/p.jpg"})
        if cls.mode == "no-credits":
            return self.reply(402, {"code": "INSUFFICIENT_CREDITS", "message": "Balance is 0", "status": 402})
        if cls.mode == "slow":
            return self.reply(202, {"generationId": "gen-1", "status": "PROCESSING"})
        if cls.mode == "fail":
            return self.reply(422, {"code": "MULTI_PRODUCT_NOT_SUPPORTED", "message": "hat", "status": 422})
        self.reply(200, {"generationId": "gen-1", "status": "COMPLETED", "resultImageUrl": self.base() + "/result.jpg"})

    def do_GET(self):
        cls = type(self)
        if self.path == "/result.jpg":
            return self.reply(200, RESULT, "image/jpeg")
        cls.calls.append((self.path, None))
        cls.polls += 1
        if cls.polls < 3:
            return self.reply(200, {"generationId": "gen-1", "status": "PROCESSING"})
        self.reply(200, {"generationId": "gen-1", "status": "COMPLETED", "resultImageUrl": self.base() + "/result.jpg"})

    def base(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def log_message(self, *args):
        pass


class GenlookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGenlook)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        FakeGenlook.calls, FakeGenlook.mode, FakeGenlook.polls = [], "ok", 0
        self.sleeps = []
        self.client = genlook.Client("gk_test", self.base, sleep=self.sleeps.append)

    def run_try_on(self, garments=((SHIRT, "image/png"),), description="寬鬆版"):
        return genlook.try_on(PERSON, "image/jpeg", list(garments), description, client=self.client)

    def test_uploads_uncropped_then_tries_on_every_garment(self):
        result = self.run_try_on([(SHIRT, "image/png"), (JEANS, "image/png")])
        self.assertEqual(result, {"image": RESULT, "media_type": "image/jpeg", "note": ""})
        (upload_path, upload), (tryon_path, form) = FakeGenlook.calls
        self.assertEqual(upload_path, "/images/upload")
        self.assertEqual(upload["file"], (PERSON, "image/jpeg"))
        self.assertEqual(upload["crop"][0], b"false")
        self.assertEqual(tryon_path, "/try-on/sync")
        data = json.loads(form["data"][0])
        self.assertEqual(data["person"], {"image": {"source": {"id": "img-1"}}})
        self.assertEqual([p["images"][0]["source"]["fileKey"] for p in data["products"]], ["product0", "product1"])
        self.assertEqual(data["products"][0]["description"], "寬鬆版")
        self.assertIs(data["mock"], False)
        self.assertEqual((form["product0"], form["product1"]), ((SHIRT, "image/png"), (JEANS, "image/png")))

    def test_polls_when_the_result_takes_longer(self):
        FakeGenlook.mode = "slow"
        self.assertEqual(self.run_try_on()["image"], RESULT)
        self.assertEqual([c[0] for c in FakeGenlook.calls[2:]], ["/generations/gen-1"] * 3)
        self.assertEqual(self.sleeps, [1, 1, 1])

    def test_errors_become_friendly_messages(self):
        for mode, message in {"no-credits": "點數用完", "fail": "單獨試穿"}.items():
            FakeGenlook.mode = mode
            with self.subTest(mode=mode), self.assertRaisesRegex(AnalysisError, message):
                self.run_try_on()
        self.client.api_key = "gk_wrong"
        with self.assertRaisesRegex(AnalysisError, "金鑰無效"):
            self.run_try_on()
        self.client.api_key = ""
        with self.assertRaisesRegex(AnalysisError, "還沒設定"):
            self.run_try_on()

    def test_needs_garment_photos(self):
        with self.assertRaisesRegex(AnalysisError, "商品照"):
            self.run_try_on(garments=[])
        with self.assertRaisesRegex(AnalysisError, "最多 4"):
            self.run_try_on(garments=[(SHIRT, "image/png")] * 5)
        self.assertEqual(FakeGenlook.calls, [])

    def test_mock_mode_is_free_and_says_so(self):
        with mock.patch.dict(os.environ, {"GENLOOK_MOCK": "1"}):
            result = self.run_try_on(description="")
        data = json.loads(FakeGenlook.calls[1][1]["data"][0])
        self.assertIs(data["mock"], True)
        self.assertNotIn("description", data["products"][0])
        self.assertIn("測試模式", result["note"])


class ProviderTests(unittest.TestCase):
    def test_genlook_when_its_key_is_set_unless_overridden(self):
        cases = [
            ({}, "gemini"),
            ({"GENLOOK_API_KEY": "gk_x"}, "genlook"),
            ({"GENLOOK_API_KEY": "gk_x", "OUTFIT_TRYON_PROVIDER": "gemini"}, "gemini"),
            ({"OUTFIT_TRYON_PROVIDER": "Genlook"}, "genlook"),
        ]
        for env, expected in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(tryon.provider(), expected)
                self.assertEqual(tryon.info()["needs_garment_photo"], expected == "genlook")


if __name__ == "__main__":
    unittest.main()
