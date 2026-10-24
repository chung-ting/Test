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
from outfit.sizing import normalize_chart, recommend

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



def raw_chart(**overrides):
    chart = {"is_size_chart": True, "chart_type": "garment", "category": "top", "unit": "cm",
             "girth_type": "flat", "stretch": False, "notes": "", "sizes": []}
    chart.update(overrides)
    return chart


def row(size, **values):
    keys = ["chest", "hip", "length", "shoulder", "sleeve", "thigh", "waist"]
    return {"size": size, **{k: values.get(k) for k in keys}}


TOP_CHART = {"category": "top", "stretch": False, "notes": "", "sizes": [
    {"size": "S", "chest": 48, "length": 66},
    {"size": "M", "chest": 51, "length": 68},
    {"size": "L", "chest": 54, "length": 70},
]}


class ChartTests(unittest.TestCase):
    def test_converts_circumference_and_inches_to_flat_cm(self):
        chart = normalize_chart(raw_chart(unit="inch", girth_type="circumference",
                                          sizes=[row("M", chest=40, length=27, waist=30)]))
        # chest is a girth: inches -> cm, then halved; length isn't; waist isn't a top measure
        self.assertEqual(chart["sizes"], [{"size": "M", "chest": 50.8, "length": 68.6}])

    def test_rejects_body_charts_and_non_charts(self):
        with self.assertRaisesRegex(AnalysisError, "身體尺寸"):
            normalize_chart(raw_chart(chart_type="body", sizes=[row("M", chest=90)]))
        with self.assertRaises(AnalysisError):
            normalize_chart(raw_chart(is_size_chart=False))

    def test_extract_chart_via_claude(self):
        from outfit.sizing import extract_chart
        raw = raw_chart(category="bottom", sizes=[row("28", waist=36, length=100, chest=99)])
        client = FakeClient(fake_response(json.dumps(raw)))
        chart = extract_chart(PNG, "image/png", client=client)
        self.assertEqual(chart["sizes"], [{"size": "28", "waist": 36, "length": 100}])
        schema = client.calls[0]["output_config"]["format"]["schema"]
        self.assertIn("thigh", schema["properties"]["sizes"]["items"]["required"])


class RecommendTests(unittest.TestCase):
    def test_picks_closest_size(self):
        rec = recommend(TOP_CHART, {"chest": 51, "length": 68.5})
        self.assertEqual(rec["recommended"], "M")
        m = next(s for s in rec["sizes"] if s["size"] == "M")
        self.assertEqual([(f["label"], f["diff"], f["verdict"]) for f in m["fits"]],
                         [("胸寬", 0.0, "剛好"), ("衣長", -0.5, "剛好")])
        self.assertEqual(rec["warnings"], [])

    def test_prefers_loose_over_tight_unless_stretchy(self):
        # Halfway between M (51) and L (54): tight is penalized more, so L wins.
        self.assertEqual(recommend(TOP_CHART, {"chest": 52.5})["recommended"], "L")
        self.assertEqual(recommend({**TOP_CHART, "stretch": True}, {"chest": 52.5})["recommended"], "M")

    def test_verdict_wording(self):
        rec = recommend(TOP_CHART, {"chest": 46, "length": 75})
        s = next(s for s in rec["sizes"] if s["size"] == "L")
        self.assertEqual([f["verdict"] for f in s["fits"]], ["明顯寬鬆", "明顯短"])

    def test_warns_when_no_size_fits(self):
        rec = recommend(TOP_CHART, {"chest": 60})
        self.assertEqual(rec["recommended"], "L")
        self.assertIn("明顯偏緊", rec["warnings"][0])
        self.assertIn("圍度", recommend(TOP_CHART, {"length": 68})["warnings"][0])

    def test_rejects_bad_input(self):
        for chart, ref in [(TOP_CHART, {}), (TOP_CHART, {"chest": -3}), (TOP_CHART, {"chest": "51"}),
                           ({**TOP_CHART, "category": "hat"}, {"chest": 51}), (TOP_CHART, {"waist": 40})]:
            with self.assertRaises(AnalysisError, msg=(chart.get("category"), ref)):
                recommend(chart, ref)

class ServerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_analyze(image, media_type, occasion, note):
            self.calls.append((image, media_type, occasion, note))
            if occasion == "bad":
                raise AnalysisError("未知的場合：bad")
            return RESULT

        handler = type("TestHandler", (Handler,), {"analyzer": staticmethod(fake_analyze),
                                                   "chart_reader": staticmethod(lambda image, mt: TOP_CHART),
                                                   "log_message": lambda *a: None})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, path="/api/analyze"):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
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

    def test_size_endpoints(self):
        with urllib.request.urlopen(self.base + "/api/measures") as r:
            self.assertEqual(json.load(r)["top"]["chest"]["label"], "胸寬")
        status, body = self.post({"chart": TOP_CHART, "reference": {"chest": 51}}, "/api/recommend-size")
        self.assertEqual((status, body["recommended"]), (200, "M"))
        status, body = self.post({"chart": {"category": "top", "sizes": "oops"}, "reference": {"chest": 51}},
                                 "/api/recommend-size")
        self.assertEqual(status, 400)
        data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
        status, body = self.post({"image": data_url}, "/api/size-chart")
        self.assertEqual((status, body), (200, TOP_CHART))

    def test_decode_data_url(self):
        self.assertEqual(decode_data_url("data:image/jpeg;base64,AAAA"), (b"\x00\x00\x00", "image/jpeg"))
        with self.assertRaises(AnalysisError):
            decode_data_url("data:image/jpeg;base64,@@@")


if __name__ == "__main__":
    unittest.main()
