"""Build a single-page 3D avatar viewer: python3 -m outfit.demo.build_viewer MODEL.glb OUT.html [--title NAME]

The GLB is embedded as base64 and parsed in the browser (no network fetch), so the page can be
published on its own. Drag to rotate, pinch or scroll to zoom; if the model carries animations
(after rigging), a button per animation plays it.
"""

import argparse
import base64
from pathlib import Path

THREE = "https://cdn.jsdelivr.net/npm/three@0.147.0"

PAGE = """<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
  /* One screen: the model fills the stage, controls sit in a strip underneath. */
  :root {
    --bg: #f3f0f5; --stage: #e9e4ee; --ink: #26212b; --muted: #6f6677; --line: #d9d2df;
    --accent: #8a4fb8; --accent-ink: #ffffff;
    --font: "PingFang TC", "Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --bg: #17141a; --stage: #211d26; --ink: #efe9f4; --muted: #a59bb0; --line: #3a3342;
      --accent: #c08ff0; --accent-ink: #17141a; color-scheme: dark; }
  }
  :root[data-theme="dark"] { --bg: #17141a; --stage: #211d26; --ink: #efe9f4; --muted: #a59bb0; --line: #3a3342;
    --accent: #c08ff0; --accent-ink: #17141a; color-scheme: dark; }
  html, body { height: 100%; }
  body { background: var(--bg); color: var(--ink); font: 15px/1.5 var(--font); margin: 0;
         padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px));
         padding-block: 12px; box-sizing: border-box; display: flex; flex-direction: column; gap: 10px; }
  h1 { margin: 0; font-size: 1.25rem; }
  #stage { position: relative; flex: 1; min-height: 320px; border-radius: 16px; background: var(--stage); overflow: hidden;
           touch-action: none; }
  #stage canvas { display: block; width: 100%; height: 100%; }
  #status { position: absolute; inset: 0; display: grid; place-items: center; color: var(--muted); }
  .row { display: flex; flex-wrap: wrap; gap: 8px; }
  button { font: inherit; border: 1px solid var(--line); background: var(--bg); color: var(--ink); border-radius: 999px;
           padding: 8px 14px; cursor: pointer; }
  button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .hint { color: var(--muted); font-size: .85rem; margin: 0; }
</style>

<h1>__TITLE__</h1>
<div id="stage"><div id="status">載入 3D 模型中…</div></div>
<div class="row" id="views">
  <button data-angle="0">正面</button><button data-angle="90">側面</button><button data-angle="180">背面</button>
  <button id="spin" aria-pressed="true">自動旋轉</button>
</div>
<div class="row" id="moves" hidden></div>
<p class="hint" id="hint">用手指拖曳旋轉，兩指縮放。這個模型還沒綁骨架，所以還不會動；綁好骨架後這裡會出現動作按鈕。</p>

<script src="__THREE__/build/three.min.js"></script>
<script src="__THREE__/examples/js/loaders/GLTFLoader.js"></script>
<script src="__THREE__/examples/js/controls/OrbitControls.js"></script>
<script>
"use strict";
const MODEL = "__MODEL__";
const stage = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.append(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 100);
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7f90, 1.1));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(1.5, 2.5, 2.5);
scene.add(key);
const rim = new THREE.DirectionalLight(0xe8dcff, 0.8);
rim.position.set(-2, 1.5, -2.5);
scene.add(rim);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.enablePan = false;
controls.autoRotate = !matchMedia("(prefers-reduced-motion: reduce)").matches;
controls.autoRotateSpeed = 1.2;
document.getElementById("spin").setAttribute("aria-pressed", String(controls.autoRotate));

let mixer = null, current = null, radius = 2;
const clock = new THREE.Clock();

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

function lookFrom(deg) {
  const t = deg * Math.PI / 180, target = controls.target;
  camera.position.set(target.x + radius * Math.sin(t), target.y, target.z + radius * Math.cos(t));
  controls.update();
}

const bytes = Uint8Array.from(atob(MODEL), (c) => c.charCodeAt(0));
new THREE.GLTFLoader().parse(bytes.buffer, "", (gltf) => {
  const model = gltf.scene;
  scene.add(model);
  const box = new THREE.Box3().setFromObject(model), size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  controls.target.copy(center);
  radius = size.y / (2 * Math.tan(camera.fov * Math.PI / 360)) * 1.25;
  controls.minDistance = radius * 0.3;
  controls.maxDistance = radius * 2.5;
  lookFrom(0);
  document.getElementById("status").remove();

  if (gltf.animations.length) {
    mixer = new THREE.AnimationMixer(model);
    const moves = document.getElementById("moves");
    const buttons = gltf.animations.map((clip, i) => {
      const b = document.createElement("button");
      b.textContent = clip.name || `動作 ${i + 1}`;
      b.setAttribute("aria-pressed", "false");
      b.onclick = () => {
        if (current) current.fadeOut(0.25);
        current = mixer.clipAction(clip).reset().fadeIn(0.25).play();
        buttons.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      };
      return b;
    });
    moves.replaceChildren(...buttons);
    moves.hidden = false;
    document.getElementById("hint").textContent = "用手指拖曳旋轉，兩指縮放，點下面的按鈕切換動作。";
    buttons[0].click();
  }
}, (err) => {
  document.getElementById("status").textContent = "3D 模型讀取失敗";
  console.error(err);
});

document.querySelectorAll("[data-angle]").forEach((b) => { b.onclick = () => lookFrom(Number(b.dataset.angle)); });
document.getElementById("spin").onclick = (e) => {
  controls.autoRotate = !controls.autoRotate;
  e.currentTarget.setAttribute("aria-pressed", String(controls.autoRotate));
};
renderer.setAnimationLoop(() => {
  if (mixer) mixer.update(clock.getDelta());
  controls.update();
  renderer.render(scene, camera);
});
</script>
"""


def build(model: bytes, title: str) -> str:
    return (PAGE.replace("__TITLE__", title).replace("__THREE__", THREE)
            .replace("__MODEL__", base64.b64encode(model).decode()))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--title", default="我的 3D 分身")
    args = parser.parse_args(argv)
    args.out.write_text(build(args.model.read_bytes(), args.title))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
