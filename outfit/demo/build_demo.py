"""Build the clickable demo of the web app: python3 -m outfit.demo.build_demo OUT.html

The demo is the real page (outfit/static/index.html) with demo_backend.js answering its /api calls
in the browser with example data, so it can be published as a single page and opened on a phone.
The body-measurement tab needs MediaPipe, which a published page can't load, so it links to the
standalone measuring page instead.
"""

import argparse
import json
import re
from pathlib import Path

from ..analyze import OCCASIONS
from ..avatar import STYLES
from ..sizing import MEASURES

HERE = Path(__file__).resolve().parent
PAGE = HERE.parent / "static" / "index.html"
MEASURE_PAGE = "https://claude.ai/artifact/GbWNf8CohBuqXPmaetYQEn"

AFTER = """<script>
(function () {
  const note = document.createElement("div");
  note.className = "card demo-note";
  note.innerHTML = "<b>示範版</b>：可以點所有功能看畫面和流程，AI 結果都是範例（標有「示範」）。"
    + "正式版部署後才會真的分析穿搭、產生試穿照和分身。";
  document.querySelector("main h1").after(note);
  const body = document.getElementById("tab-body");
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML = "<h2>身體尺寸</h2><p>這個功能已經有可以直接用的版本，照片只在手機上分析。</p>"
    + '<a class="btn btn-primary" style="display:block;text-align:center;text-decoration:none" href="__MEASURE__" target="_blank" rel="noopener">打開「照片量身」</a>';
  body.replaceChildren(card);
})();
</script>
""".replace("__MEASURE__", MEASURE_PAGE)


def build(page: str) -> str:
    # The artifact host supplies the document skeleton.
    page = re.sub(r"<!doctype html>\s*|<html[^>]*>\s*|</?head>\s*|</?body>\s*|</html>\s*", "", page)
    page = re.sub(r'<meta (charset|name="viewport")[^>]*>\s*', "", page)
    page = page.replace("<title>穿搭分析</title>", "<title>穿搭換裝 App</title>")

    # Dark mode that also follows an explicit light/dark choice from the viewer.
    dark = re.search(r"@media \(prefers-color-scheme: dark\) \{\s*:root \{(.*?)\}\s*\}", page, re.S)
    tokens = dark.group(1).strip()
    page = page.replace(dark.group(0), (
        f'@media (prefers-color-scheme: dark) {{\n    :root:not([data-theme="light"]) {{ {tokens} color-scheme: dark; }}\n  }}\n'
        f'  :root[data-theme="dark"] {{ {tokens} color-scheme: dark; }}\n'
        "  .demo-note { border-color: var(--accent); font-size: .92rem; }"))

    data = {
        "occasions": OCCASIONS,
        "measures": {cat: {k: {"label": m.label, "hint": m.hint} for k, m in ms.items()} for cat, ms in MEASURES.items()},
        "styles": {k: label for k, (label, _) in STYLES.items()},
    }
    backend = (HERE / "demo_backend.js").read_text()
    page = page.replace('<script src="/static/keepface.js"></script>',
                        f"<script>window.DEMO_DATA = {json.dumps(data, ensure_ascii=False)};</script>\n"
                        f"<script>\n{backend}</script>")
    return page.rstrip() + "\n" + AFTER


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    args.out.write_text(build(PAGE.read_text()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
