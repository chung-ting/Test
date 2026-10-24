import base64
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from outfit import AnalysisError, analyze
from outfit.__main__ import Handler, decode_data_url
from outfit.analyze import build_request

PNG = b"\x89PNG\r\n\x1a\nfake"
RESULT = {
    "is_person": True,
    "summary": "整體清爽",
    "current_outfit": {"description": "白T配牛仔褲", "score": 12, "strengths": ["簡潔"], "improvements": ["加外套"]},
    "body_and_color": {"body_notes": "肩線平直", "flattering_colors": [{"name": "海軍藍", "hex": "#1f2a44"}],
                       "colors_to_avoid": []},
    "suggestions": [{"title": "俐落通勤", "items": [{"category": "上衣", "item": "牛津襯衫", "color": "淺藍"}],
                     "why": "提升正式感"}],
}


def fake_response(text=None, stop_reason="end_turn"):
    content = [SimpleNamespace(type="text", text=text)] if text is not None else []
    return SimpleNamespace(stop_reason=stop_reason, content=content)


class FakeClient:
    def __init__(self, response):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class AnalyzeTests(unittest.TestCase):
    def test_request_carries_image_occasion_and_note(self):
        req = build_request(PNG, "image/png", "work", "想顯高")
        image, text = req["messages"][0]["content"]
        self.assertEqual(base64.b64decode(image["source"]["data"]), PNG)
        self.assertEqual(image["source"]["media_type"], "image/png")
        self.assertIn("上班通勤", text["text"])
        self.assertIn("想顯高", text["text"])
        self.assertEqual(req["output_config"]["format"]["type"], "json_schema")

    def test_rejects_bad_media_type_and_occasion(self):
        with self.assertRaises(AnalysisError):
            build_request(PNG, "application/pdf", "daily")
        with self.assertRaises(AnalysisError):
            build_request(PNG, "image/png", "moon")

    def test_parses_result_and_clamps_score(self):
        client = FakeClient(fake_response(json.dumps(RESULT)))
        result = analyze(PNG, "image/png", "daily", client=client)
        self.assertEqual(result["current_outfit"]["score"], 10)
        self.assertEqual(result["suggestions"][0]["title"], "俐落通勤")
        self.assertEqual(len(client.calls), 1)

    def test_refusal_and_truncation_raise_friendly_errors(self):
        for resp in (fake_response(stop_reason="refusal"), fake_response("{", stop_reason="max_tokens")):
            with self.assertRaises(AnalysisError):
                analyze(PNG, "image/png", "daily", client=FakeClient(resp))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_analyze(image, media_type, occasion, note):
            self.calls.append((image, media_type, occasion, note))
            if occasion == "bad":
                raise AnalysisError("未知的場合：bad")
            return RESULT

        handler = type("TestHandler", (Handler,), {"analyzer": staticmethod(fake_analyze),
                                                   "log_message": lambda *a: None})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body):
        req = urllib.request.Request(self.base + "/api/analyze", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def test_serves_page_and_occasions(self):
        with urllib.request.urlopen(self.base + "/") as r:
            self.assertIn("穿搭分析", r.read().decode())
        with urllib.request.urlopen(self.base + "/api/occasions") as r:
            self.assertIn("work", json.load(r))

    def test_analyze_round_trip(self):
        data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        status, body = self.post({"image": data_url, "occasion": "date", "note": "hi"})
        self.assertEqual(status, 200)
        self.assertEqual(body["summary"], "整體清爽")
        self.assertEqual(self.calls, [(PNG, "image/png", "date", "hi")])

    def test_bad_input_returns_400(self):
        self.assertEqual(self.post({"image": "not-a-data-url"})[0], 400)
        data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        status, body = self.post({"image": data_url, "occasion": "bad"})
        self.assertEqual((status, body["error"]), (400, "未知的場合：bad"))

    def test_decode_data_url(self):
        self.assertEqual(decode_data_url("data:image/jpeg;base64,AAAA"), (b"\x00\x00\x00", "image/jpeg"))
        with self.assertRaises(AnalysisError):
            decode_data_url("data:image/jpeg;base64,@@@")


if __name__ == "__main__":
    unittest.main()
