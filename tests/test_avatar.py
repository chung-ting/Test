import base64
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from google import genai
from google.genai import types

from outfit import AnalysisError, avatar
from tests.test_tryon import RESULT, FakeGemini, image_reply

PHOTO = b"\xff\xd8person-jpeg"
SHOES = b"\x89PNGshoes"


def sent_images(body):
    return [base64.urlsafe_b64decode(p["inlineData"]["data"] + "==") for p in body["contents"][0]["parts"][1:]]


class AvatarTests(unittest.TestCase):
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
        self.tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"OUTFIT_DATA_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_each_style_has_its_own_prompt_and_fixed_pose(self):
        prompts = []
        for style in avatar.STYLES:
            result = avatar.create_avatar(PHOTO, "image/jpeg", style, client=self.client)
            self.assertEqual(result["image"], RESULT)
            body = FakeGemini.requests[-1][1]
            prompts.append(body["contents"][0]["parts"][0]["text"])
            self.assertEqual(sent_images(body), [PHOTO])
            self.assertEqual(body["generationConfig"]["imageConfig"], {"aspectRatio": "3:4"})
        self.assertEqual(len(set(prompts)), 3)
        self.assertTrue(all("standing upright facing the camera" in p for p in prompts))
        with self.assertRaisesRegex(AnalysisError, "風格"):
            avatar.create_avatar(PHOTO, "image/jpeg", "watercolor", client=self.client)

    def test_dressing_sends_the_avatar_first_then_the_garments(self):
        avatar.dress_avatar(RESULT, "image/png", "game", [(SHOES, "image/png")], "白色帆布鞋", client=self.client)
        body = FakeGemini.requests[-1][1]
        prompt = body["contents"][0]["parts"][0]["text"]
        self.assertIn("same character", prompt)
        self.assertIn(avatar.STYLES["game"][1], prompt)
        self.assertIn("白色帆布鞋", prompt)
        self.assertEqual(sent_images(body), [RESULT, SHOES])
        with self.assertRaisesRegex(AnalysisError, "衣服照片"):
            avatar.dress_avatar(RESULT, "image/png", "game", [], " ", client=self.client)

    def test_saves_one_avatar_on_this_computer(self):
        self.assertIsNone(avatar.load_avatar())
        avatar.save_avatar(b"first", "image/png", "figure")
        avatar.save_avatar(b"second", "image/jpeg", "game")
        saved = avatar.load_avatar()
        self.assertEqual((saved["image"], saved["style"], saved["media_type"]), (b"second", "game", "image/jpeg"))
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()), ["avatar.jpg", "avatar.json"])

    def test_command_line_writes_one_file_per_style(self):
        photo = Path(self.tmp.name) / "me.jpg"
        photo.write_bytes(PHOTO)
        out = Path(self.tmp.name) / "out"
        fake = lambda contents, aspect, client=None: {"image": RESULT, "media_type": "image/png", "note": ""}
        with mock.patch("outfit.avatar.generate_image", fake), redirect_stdout(io.StringIO()) as printed:
            avatar.main([str(photo), "-o", str(out)])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["me-animation.png", "me-figure.png", "me-game.png"])
        self.assertIn("Q 版 3D 公仔", printed.getvalue())


class AvatarRouteTests(unittest.TestCase):
    def test_preview_save_load_and_dress(self):
        from outfit.__main__ import Handler

        dressed = []
        handler = type("H", (Handler,), {
            "avatar_maker": staticmethod(lambda img, mt, style: {"image": f"{style}-avatar".encode(), "media_type": "image/png"}),
            "avatar_dresser": staticmethod(lambda img, mt, style, garments, desc: dressed.append((img, style, garments, desc))
                                           or {"image": b"dressed", "media_type": "image/png"}),
            "log_message": lambda *a: None,
        })
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        url = lambda data, mt: f"data:{mt};base64," + base64.b64encode(data).decode()

        def call(path, body=None):
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req) as r:
                    return r.status, json.load(r)
            except urllib.error.HTTPError as e:
                return e.code, json.load(e)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OUTFIT_DATA_DIR": tmp}):
            try:
                status, info = call("/api/avatar")
                self.assertEqual((info["avatar"], info["styles"]["figure"]), (None, "Q 版 3D 公仔"))
                self.assertIn("先產生", call("/api/avatar/dress", {"description": "鞋子"})[1]["error"])
                _, preview = call("/api/avatar/preview", {"image": url(PHOTO, "image/jpeg"), "style": "figure"})
                self.assertEqual(preview["image"], url(b"figure-avatar", "image/png"))
                self.assertEqual(call("/api/avatar/save", {"image": preview["image"], "style": "nope"})[0], 400)
                self.assertEqual(call("/api/avatar/save", {"image": preview["image"], "style": "figure"}), (200, {"saved": True}))
                self.assertEqual(call("/api/avatar")[1]["avatar"], {"image": preview["image"], "style": "figure"})
                _, look = call("/api/avatar/dress", {"garments": [url(SHOES, "image/png")], "description": "鞋子"})
                self.assertEqual(look["image"], url(b"dressed", "image/png"))
                self.assertEqual(dressed, [(b"figure-avatar", "figure", [(SHOES, "image/png")], "鞋子")])
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
