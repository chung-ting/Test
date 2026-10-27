import base64
import importlib.util
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
        patcher = mock.patch.dict(os.environ, {"OUTFIT_DATA_DIR": self.tmp.name, "OUTFIT_AVATAR_PROVIDER": "gemini"})
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
            self.assertEqual(body["generationConfig"]["imageConfig"], {"aspectRatio": "2:3"})
        self.assertEqual(len(set(prompts)), len(avatar.STYLES))
        self.assertTrue(all("head to toe" in p or "from the top of the head down to the shoes" in p for p in prompts))
        self.assertTrue(all("do not lighten" in p or "Copy the hair color and the skin color" in p for p in prompts))
        # Naming example items makes the model draw them, so the 2D prompt names none.
        self.assertTrue(all(word not in prompts[0] for word in ("glasses", "freckles", "braces", "jewelry")))
        self.assertIn("2D animated character", prompts[0])  # film2d comes first and is the default
        with self.assertRaisesRegex(AnalysisError, "風格"):
            avatar.create_avatar(PHOTO, "image/jpeg", "watercolor", client=self.client)

    def test_dressing_sends_the_avatar_first_then_the_garments(self):
        avatar.dress_avatar(RESULT, "image/png", "fashion", [(SHOES, "image/png")], "白色帆布鞋", client=self.client)
        body = FakeGemini.requests[-1][1]
        prompt = body["contents"][0]["parts"][0]["text"]
        self.assertIn("same character", prompt)
        self.assertIn(avatar.STYLES["fashion"][1], prompt)
        self.assertIn("白色帆布鞋", prompt)
        self.assertEqual(sent_images(body), [RESULT, SHOES])
        with self.assertRaisesRegex(AnalysisError, "衣服照片"):
            avatar.dress_avatar(RESULT, "image/png", "fashion", [], " ", client=self.client)

    def test_saves_one_avatar_on_this_computer(self):
        self.assertIsNone(avatar.load_avatar())
        avatar.save_avatar(b"first", "image/png", "figure")
        avatar.save_avatar(b"second", "image/jpeg", "fashion")
        saved = avatar.load_avatar()
        self.assertEqual((saved["image"], saved["style"], saved["media_type"]), (b"second", "fashion", "image/jpeg"))
        self.assertEqual(sorted(p.name for p in Path(self.tmp.name).iterdir()), ["avatar.jpg", "avatar.json"])

    def test_command_line_writes_one_file_per_style(self):
        photo = Path(self.tmp.name) / "me.jpg"
        photo.write_bytes(PHOTO)
        out = Path(self.tmp.name) / "out"
        fake = lambda contents, aspect, client=None: {"image": RESULT, "media_type": "image/png", "note": ""}
        with mock.patch("outfit.avatar.generate_image", fake), redirect_stdout(io.StringIO()) as printed:
            avatar.main([str(photo), "-o", str(out)])
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["me-fashion.png", "me-figure.png", "me-film2d.png", "me-game.png"])
        self.assertIn("2D 動畫電影風", printed.getvalue())
        with mock.patch("outfit.avatar.generate_image", fake), redirect_stdout(io.StringIO()):
            avatar.main([str(photo), "-o", str(out), "--template", "all"])
        self.assertEqual(len(list(out.glob("me-template-*.png"))), len(avatar.TEMPLATES))

    def test_templates_dress_the_2d_character_in_their_outfit(self):
        for key, (_, _, outfit, _scene) in avatar.TEMPLATES.items():
            avatar.try_template(PHOTO, "image/jpeg", key, client=self.client)
            prompt = FakeGemini.requests[-1][1]["contents"][0]["parts"][0]["text"]
            self.assertIn("hand-drawn 2D animated character", prompt)
            self.assertIn(outfit, prompt)
            self.assertIn("#9EE6B8", prompt)  # flat key color, cut out afterwards
            self.assertNotIn("Image 2", prompt)

    def test_style_reference_is_image_two_and_only_steers_the_style(self):
        avatar.create_avatar(PHOTO, "image/jpeg", "film2d", client=self.client, style_ref=(SHOES, "image/png"))
        body = FakeGemini.requests[-1][1]
        prompt = body["contents"][0]["parts"][0]["text"]
        self.assertIn("using Image 2 only as a reference for the illustration style", prompt)
        self.assertIn("Do not copy the outfit or identity from Image 2", prompt)
        self.assertEqual(sent_images(body), [PHOTO, SHOES])
        avatar.create_avatar(PHOTO, "image/jpeg", "figure", client=self.client, style_ref=(SHOES, "image/png"))
        self.assertEqual(sent_images(FakeGemini.requests[-1][1]), [PHOTO])  # other styles ignore it
        with self.assertRaisesRegex(AnalysisError, "格式"):
            avatar.create_avatar(PHOTO, "image/jpeg", "film2d", client=self.client, style_ref=(SHOES, "image/bmp"))

    def test_transparent_avatars_go_back_on_the_key_color_for_dressing(self):
        from PIL import Image

        cutout = Image.new("RGBA", (40, 60), (0, 0, 0, 0))
        cutout.paste((200, 30, 60, 255), (10, 10, 30, 50))
        buf = io.BytesIO()
        cutout.save(buf, "PNG")
        flat, media_type = avatar.on_key_color(buf.getvalue(), "image/png")
        img = Image.open(io.BytesIO(flat))
        self.assertEqual((img.mode, media_type, img.getpixel((0, 0)), img.getpixel((20, 30))),
                         ("RGB", "image/png", avatar.KEY_COLOR, (200, 30, 60)))
        self.assertEqual(avatar.on_key_color(PHOTO, "image/jpeg"), (PHOTO, "image/jpeg"))
        with self.assertRaisesRegex(AnalysisError, "模板"):
            avatar.try_template(PHOTO, "image/jpeg", "space", client=self.client)

    def test_close_ups_get_room_for_the_whole_body(self):
        from PIL import Image

        def jpeg(w, h):
            out = io.BytesIO()
            Image.new("RGB", (w, h), (90, 120, 160)).save(out, "JPEG")
            return out.getvalue()

        selfie = jpeg(1206, 1300)
        framed, media_type = avatar.frame_for_full_body(selfie, "image/png")
        img = Image.open(io.BytesIO(framed))
        self.assertEqual((img.size, media_type), ((896, 1344), "image/jpeg"))
        # empty space left for the legs and feet (JPEG may shift the color by a step or two)
        self.assertTrue(all(abs(a - b) <= 3 for a, b in zip(img.getpixel((448, 1300)), (246, 228, 234))))
        tall = jpeg(600, 1100)
        self.assertEqual(avatar.frame_for_full_body(tall, "image/jpeg"), (tall, "image/jpeg"))
        self.assertEqual(avatar.frame_for_full_body(PHOTO, "image/jpeg"), (PHOTO, "image/jpeg"))


@unittest.skipUnless(importlib.util.find_spec("rembg"), "rembg not installed")
class BackgroundRemovalTests(unittest.TestCase):
    """The cut-out itself was checked on real generated characters; here, how we call rembg."""

    def test_returns_a_png_cut_out_with_the_anime_model(self):
        from PIL import Image

        img = Image.new("RGB", (40, 60), avatar.KEY_COLOR)
        buf = io.BytesIO()
        img.save(buf, "JPEG")
        cut = Image.new("RGBA", (40, 60), (0, 0, 0, 0))
        with mock.patch("rembg.new_session", return_value="session") as session, \
                mock.patch("rembg.remove", return_value=cut) as remove, \
                mock.patch.object(avatar, "_rembg_session", None):
            result = avatar.remove_background({"image": buf.getvalue(), "media_type": "image/jpeg", "note": ""})
            avatar.remove_background({"image": buf.getvalue(), "media_type": "image/jpeg", "note": ""})
        session.assert_called_once_with("isnet-anime")  # loaded once, then reused
        self.assertEqual(remove.call_args.kwargs, {"session": "session", "post_process_mask": True})
        self.assertEqual(result["media_type"], "image/png")
        self.assertEqual(Image.open(io.BytesIO(result["image"])).mode, "RGBA")
        self.assertEqual(avatar.remove_background({"image": PHOTO, "media_type": "image/jpeg"})["image"], PHOTO)

    def test_trims_empty_space_around_the_character(self):
        from PIL import Image

        cut = Image.new("RGBA", (400, 600), (0, 0, 0, 0))
        cut.paste((10, 20, 30, 255), (150, 100, 250, 500))  # 100 x 400 character
        trimmed = avatar.trim_to_character(cut)
        self.assertEqual(trimmed.size, (100 + 2 * 16, 400 + 2 * 16))  # 4% of the longer side on each edge
        empty = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        self.assertEqual(avatar.trim_to_character(empty).size, (10, 10))


class ProviderTests(unittest.TestCase):
    def test_qwen_when_a_hugging_face_token_is_set_unless_overridden(self):
        cases = [({}, "gemini", 4), ({"HF_TOKEN": "hf_x"}, "qwen", 2),
                 ({"HF_TOKEN": "hf_x", "OUTFIT_AVATAR_PROVIDER": "gemini"}, "gemini", 4),
                 ({"OUTFIT_AVATAR_PROVIDER": "FLUX"}, "flux", 4)]
        for env, expected, limit in cases:
            with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual((avatar.provider(), avatar.dress_limit()), (expected, limit))

    def space_result(self, tmp, gallery):
        out = Path(tmp) / "image.webp"
        out.write_bytes(b"RIFFspacewebp")
        client = mock.MagicMock()
        client.predict.return_value = ([{"image": {"path": str(out)}, "caption": None}] if gallery
                                       else {"path": str(out)}, 42)
        return client

    def test_qwen_space_gets_every_image_and_our_prompt_as_is(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"OUTFIT_AVATAR_PROVIDER": "qwen", "HF_TOKEN": "hf_x"}), \
                mock.patch("gradio_client.Client") as make:
            client = make.return_value = self.space_result(tmp, gallery=True)
            result = avatar.try_template(PHOTO, "image/jpeg", "campus")
            self.assertEqual((result["image"], result["media_type"]), (b"RIFFspacewebp", "image/webp"))
            self.assertEqual(make.call_args.args[0], avatar.QWEN_SPACE)
            gallery, prompt, *rest = client.predict.call_args.args
            self.assertEqual(len(gallery), 1)
            self.assertIn("pleated skirt", prompt)
            self.assertEqual(rest[-3:], [1248, 832, False])  # height, width, no prompt rewriting
            avatar.dress_avatar(b"avatar", "image/png", "game", [(SHOES, "image/png"), (SHOES, "image/png")], "")
            self.assertEqual(len(client.predict.call_args.args[0]), 3)
            with self.assertRaisesRegex(AnalysisError, "最多 2 件"):
                avatar.dress_avatar(b"avatar", "image/png", "game", [(SHOES, "image/png")] * 3, "")

    def test_flux_space_results_and_quota_errors(self):
        from gradio_client.exceptions import AppError

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.webp"
            out.write_bytes(b"RIFFfluxwebp")
            client = mock.MagicMock()
            client.predict.return_value = ({"path": str(out)}, 42)
            with mock.patch.dict(os.environ, {"OUTFIT_AVATAR_PROVIDER": "flux", "HF_TOKEN": "hf_x"}), \
                    mock.patch("gradio_client.Client", return_value=client) as make:
                result = avatar.try_template(PHOTO, "image/jpeg", "city")
                self.assertEqual((result["image"], result["media_type"]), (b"RIFFfluxwebp", "image/webp"))
                self.assertEqual(make.call_args.kwargs["token"], "hf_x")
                self.assertIn("street-style", client.predict.call_args.args[1])
                client.predict.side_effect = AppError("You have exceeded your ZeroGPU quota")
                with self.assertRaisesRegex(AnalysisError, "免費產圖次數用完"):
                    avatar.try_template(PHOTO, "image/jpeg", "city")
                client.predict.side_effect = ConnectionError("down")
                with self.assertRaisesRegex(AnalysisError, "連不上"):
                    avatar.try_template(PHOTO, "image/jpeg", "city")


class AvatarRouteTests(unittest.TestCase):
    def test_preview_save_load_and_dress(self):
        from outfit.__main__ import Handler

        dressed, refs = [], []
        handler = type("H", (Handler,), {
            "avatar_maker": staticmethod(lambda img, mt, style, style_ref=None:
                                         {"image": f"{style}-avatar".encode(), "media_type": "image/png"}),
            "template_maker": staticmethod(lambda img, mt, key, style_ref=None:
                                           refs.append(style_ref) or {"image": f"{key}-look".encode(), "media_type": "image/webp"}),
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
                self.assertEqual((info["avatar"], list(info["styles"])[0]), (None, "film2d"))
                self.assertEqual(info["templates"]["city"], {"label": "城市街拍", "icon": "🏙️"})
                self.assertIn(info["max_garments"], (2, 4))
                _, look = call("/api/avatar/template", {"image": url(PHOTO, "image/jpeg"), "template": "city"})
                self.assertEqual(look["image"], url(b"city-look", "image/webp"))
                call("/api/avatar/template", {"image": url(PHOTO, "image/jpeg"), "template": "city",
                                              "style_ref": url(SHOES, "image/png")})
                self.assertEqual(refs, [None, (SHOES, "image/png")])
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
