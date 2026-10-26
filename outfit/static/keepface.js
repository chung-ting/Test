// Put the person's original face and hair back onto an AI-edited photo.
//
// Image models redraw the whole picture, so the face drifts even when told not to. For each photo,
// MediaPipe's pose model locates the head, and the multiclass selfie segmenter (made for close-ups,
// hence the crop) finds face-skin and hair in it. The original photo is scaled and moved so its
// head and shoulder landmarks line up with the edited photo's, then pasted over both heads (so any
// extra hair or face the model drew is covered by the original background) with a soft edge.
(function () {
  "use strict";

  const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
  const SEGMENT_MODEL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
  const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
  const MAX_CROP = 512;
  const HAIR = 1, FACE = 3;
  const MAX_ASPECT_DIFF = 0.06;  // beyond this the two photos aren't framed the same way
  const MIN_FACE_SHARE = 0.0004; // face-skin pixels as a share of the image, below which no face counts as found

  let modelsPromise = null;

  function getModels() {
    if (!modelsPromise) {
      modelsPromise = (async () => {
        const { ImageSegmenter, PoseLandmarker, FilesetResolver } = await import(`${MP}/vision_bundle.mjs`);
        const vision = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
        const create = async (Task, extra) => {
          const options = (delegate) => ({ baseOptions: { delegate, ...extra.baseOptions }, runningMode: "IMAGE", ...extra.rest });
          try { return await Task.createFromOptions(vision, options("GPU")); }
          catch { return await Task.createFromOptions(vision, options("CPU")); }
        };
        const [segmenter, pose] = await Promise.all([
          create(ImageSegmenter, { baseOptions: { modelAssetPath: SEGMENT_MODEL },
                                   rest: { outputCategoryMask: true, outputConfidenceMasks: false } }),
          create(PoseLandmarker, { baseOptions: { modelAssetPath: POSE_MODEL }, rest: { numPoses: 1 } }),
        ]);
        return { segmenter, pose };
      })();
      modelsPromise.catch(() => { modelsPromise = null; });
    }
    return modelsPromise;
  }

  // Head and shoulder landmarks (pose indices 0-12) in image pixels, or null when no person is found.
  function headPoints(pose, canvas) {
    const found = pose.detect(canvas).landmarks[0];
    return found ? found.slice(0, 13).map((l) => [l.x * canvas.width, l.y * canvas.height]) : null;
  }

  // A square around the head from the landmarks (nose, eyes, ears, mouth, shoulders), or the whole image.
  function headBox(p, canvas) {
    const W = canvas.width, H = canvas.height;
    if (!p) return { x: 0, y: 0, w: W, h: H };
    const head = p.slice(0, 11);
    const cx = head.reduce((s, q) => s + q[0], 0) / head.length;
    const cy = head.reduce((s, q) => s + q[1], 0) / head.length;
    const dist = (a, b) => Math.hypot(p[a][0] - p[b][0], p[a][1] - p[b][1]);
    const side = Math.min(Math.max(dist(7, 8) * 3.2, dist(11, 12) * 1.3, 48), W, H);
    const x = Math.min(Math.max(cx - side / 2, 0), W - side);
    const y = Math.min(Math.max(cy - side * 0.6, 0), H - side);
    return { x, y, w: side, h: side };
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("image failed to load"));
      img.src = src;
    });
  }

  function canvasOf(img, w, h) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    c.getContext("2d").drawImage(img, 0, 0, w, h);
    return c;
  }

  // The head (face + hair) as an alpha mask placed at `box`, plus the face-skin bounding box in image pixels.
  function findHead({ segmenter, pose }, canvas) {
    const points = headPoints(pose, canvas);
    const box = headBox(points, canvas);
    const k = Math.min(1, MAX_CROP / box.w);
    const crop = document.createElement("canvas");
    crop.width = Math.round(box.w * k);
    crop.height = Math.round(box.h * k);
    crop.getContext("2d").drawImage(canvas, box.x, box.y, box.w, box.h, 0, 0, crop.width, crop.height);
    const mask = segmenter.segment(crop).categoryMask;
    try {
      const w = mask.width, h = mask.height, labels = mask.getAsUint8Array();
      const pixels = new ImageData(w, h);
      let minX = w, minY = h, maxX = -1, maxY = -1, faceCount = 0;
      for (let i = 0; i < labels.length; i++) {
        const label = labels[i];
        if (label !== HAIR && label !== FACE) continue;
        pixels.data[i * 4 + 3] = 255;
        if (label === FACE) {
          const x = i % w, y = (i - x) / w;
          faceCount++;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
      if (faceCount < labels.length * MIN_FACE_SHARE * (box.w * box.h) / (canvas.width * canvas.height)) return null;
      const maskCanvas = document.createElement("canvas");
      maskCanvas.width = w;
      maskCanvas.height = h;
      maskCanvas.getContext("2d").putImageData(pixels, 0, 0);
      // Mask pixels map onto the crop box; express the face box in image pixels.
      const sx = box.w / w, sy = box.h / h;
      return {
        mask: maskCanvas,
        box,
        points,
        face: { cx: box.x + (minX + maxX + 1) / 2 * sx, cy: box.y + (minY + maxY + 1) / 2 * sy, h: (maxY - minY + 1) * sy },
      };
    } finally {
      mask.close();
    }
  }

  // The scale s and shift (tx, ty) that best map points a onto points b.
  function fitScaleShift(a, b) {
    const mean = (ps) => ps.reduce((m, p) => [m[0] + p[0] / ps.length, m[1] + p[1] / ps.length], [0, 0]);
    const [ax, ay] = mean(a), [bx, by] = mean(b);
    let num = 0, den = 0;
    for (let i = 0; i < a.length; i++) {
      const dx = a[i][0] - ax, dy = a[i][1] - ay;
      num += dx * (b[i][0] - bx) + dy * (b[i][1] - by);
      den += dx * dx + dy * dy;
    }
    const s = den ? num / den : 1;
    return { s, tx: bx - s * ax, ty: by - s * ay };
  }

  // Returns { image: dataURL } with the original head restored, or { reason } when it isn't safe to.
  async function keepOriginalHead(originalSrc, editedSrc) {
    const [orig, edited] = await Promise.all([loadImage(originalSrc), loadImage(editedSrc)]);
    const W = orig.naturalWidth, H = orig.naturalHeight;
    if (Math.abs(Math.log((edited.naturalWidth / edited.naturalHeight) / (W / H))) > MAX_ASPECT_DIFF) {
      return { reason: "AI 產生的照片構圖和原圖不同，無法貼回原本的臉。" };
    }
    const o = canvasOf(orig, W, H), e = canvasOf(edited, W, H);
    const models = await getModels();
    const before = findHead(models, o), after = findHead(models, e);
    if (!before || !after) return { reason: "照片裡找不到清楚的臉，無法貼回原本的臉。" };

    // Scale and move the original photo onto the edited one: a least-squares fit of the landmarks,
    // or the face boxes when either photo has no landmarks.
    let s, tx, ty;
    if (before.points && after.points) {
      ({ s, tx, ty } = fitScaleShift(before.points, after.points));
    } else {
      s = after.face.h / before.face.h;
      tx = after.face.cx - s * before.face.cx;
      ty = after.face.cy - s * before.face.cy;
    }
    const moved = Math.hypot(after.face.cx - before.face.cx, after.face.cy - before.face.cy);
    if (s < 0.8 || s > 1.25 || moved > before.face.h * 0.6) {
      return { reason: "AI 改變了姿勢或位置太多，無法貼回原本的臉。" };
    }

    // Both heads, grown by the feather width so the soft edge falls outside them rather than across them.
    const feather = Math.max(2, Math.round(before.face.h * 0.05));
    const grown = document.createElement("canvas");
    grown.width = W;
    grown.height = H;
    const gctx = grown.getContext("2d");
    for (const [dx, dy] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7]]) {
      const ox = dx * feather, oy = dy * feather;
      gctx.setTransform(1, 0, 0, 1, ox, oy);
      gctx.drawImage(after.mask, after.box.x, after.box.y, after.box.w, after.box.h);
      gctx.setTransform(s, 0, 0, s, tx + ox, ty + oy);
      gctx.drawImage(before.mask, before.box.x, before.box.y, before.box.w, before.box.h);
    }
    const softMask = document.createElement("canvas");
    softMask.width = W;
    softMask.height = H;
    const mctx = softMask.getContext("2d");
    mctx.filter = `blur(${feather}px)`;
    mctx.drawImage(grown, 0, 0);

    const head = document.createElement("canvas");
    head.width = W;
    head.height = H;
    const hctx = head.getContext("2d");
    hctx.setTransform(s, 0, 0, s, tx, ty);
    hctx.drawImage(o, 0, 0);
    hctx.setTransform(1, 0, 0, 1, 0, 0);
    hctx.globalCompositeOperation = "destination-in";
    hctx.drawImage(softMask, 0, 0);

    e.getContext("2d").drawImage(head, 0, 0);
    return { image: e.toDataURL("image/jpeg", 0.92), transform: { s, tx, ty } };
  }

  window.KeepFace = { keepOriginalHead };
})();
