// Demo build only: answers the page's /api calls in the browser with example data, so the whole app
// can be clicked through on a phone without the server or any API key. Every AI result is an
// illustration marked "示範", not a real analysis or try-on.
(function () {
  "use strict";
  const DATA = window.DEMO_DATA; // occasions, measures and avatar styles, filled in by build_demo.py
  let savedAvatar = null;

  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const json = (data, status = 200) =>
    new Response(JSON.stringify(data), { status, headers: { "Content-Type": "application/json" } });
  const load = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });

  function stamp(ctx, w, h, text) {
    const size = Math.max(14, Math.round(w / 22));
    ctx.font = `700 ${size}px sans-serif`;
    const pad = size * 0.6, tw = ctx.measureText(text).width;
    ctx.fillStyle = "rgba(43, 39, 36, .82)";
    ctx.fillRect(0, h - size - pad * 2, Math.min(w, tw + pad * 2), size + pad * 2);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, pad, h - pad - size * 0.15);
  }

  // The person's own photo with the first garment laid over the torso: shows where the result
  // goes, clearly labelled as a demo.
  async function demoTryOn(photo, garments) {
    const img = await load(photo);
    const c = document.createElement("canvas");
    c.width = img.naturalWidth;
    c.height = img.naturalHeight;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    if (garments[0]) {
      const g = await load(garments[0]);
      const w = c.width * 0.34, h = w * g.naturalHeight / g.naturalWidth;
      ctx.globalAlpha = 0.85;
      ctx.drawImage(g, (c.width - w) / 2, c.height * 0.3, w, Math.min(h, c.height * 0.45));
      ctx.globalAlpha = 1;
    }
    stamp(ctx, c.width, c.height, "示範結果｜正式版會產生真的穿上的照片");
    return c.toDataURL("image/jpeg", 0.9);
  }

  const STYLE_LOOK = {
    game: { bg: ["#e6eefc", "#fde8ef"], head: 0.1, body: 0.55 },
    fashion: { bg: ["#fde2ea", "#e9e3ff"], head: 0.12, body: 0.5 },
    figure: { bg: ["#fff1d6", "#ffd9e4"], head: 0.2, body: 0.32 },
    animation: { bg: ["#dff3ff", "#fef3e2"], head: 0.14, body: 0.46 },
  };

  // A simple drawn character standing in for the generated avatar.
  function demoAvatar(style, outfitColor = "#8fb3e8", garment = null, label = null) {
    const look = STYLE_LOOK[style] || STYLE_LOOK.fashion;
    const W = 480, H = 640;
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const ctx = c.getContext("2d");
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, look.bg[0]);
    bg.addColorStop(1, look.bg[1]);
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);
    const cx = W / 2, headR = H * look.head / 2, top = H * 0.12;
    const bodyTop = top + headR * 2 - 4, bodyH = H * look.body;
    ctx.fillStyle = "#4a3a33"; // hair
    ctx.beginPath(); ctx.arc(cx, top + headR, headR * 1.08, Math.PI, 0); ctx.fill();
    ctx.fillStyle = "#f2c9a8"; // face
    ctx.beginPath(); ctx.arc(cx, top + headR * 1.1, headR, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#3b2d2a"; // eyes
    for (const dx of [-0.35, 0.35]) { ctx.beginPath(); ctx.ellipse(cx + dx * headR, top + headR * 1.15, headR * 0.1, headR * 0.15, 0, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = "rgba(240, 120, 140, .45)"; // cheeks
    for (const dx of [-0.55, 0.55]) { ctx.beginPath(); ctx.arc(cx + dx * headR, top + headR * 1.45, headR * 0.14, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = outfitColor; // outfit
    ctx.beginPath();
    ctx.moveTo(cx - W * 0.13, bodyTop); ctx.lineTo(cx + W * 0.13, bodyTop);
    ctx.lineTo(cx + W * 0.18, bodyTop + bodyH * 0.6); ctx.lineTo(cx - W * 0.18, bodyTop + bodyH * 0.6);
    ctx.closePath(); ctx.fill();
    if (garment) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(cx - W * 0.13, bodyTop); ctx.lineTo(cx + W * 0.13, bodyTop);
      ctx.lineTo(cx + W * 0.18, bodyTop + bodyH * 0.6); ctx.lineTo(cx - W * 0.18, bodyTop + bodyH * 0.6);
      ctx.closePath(); ctx.clip();
      ctx.drawImage(garment, cx - W * 0.18, bodyTop, W * 0.36, bodyH * 0.6);
      ctx.restore();
    }
    ctx.fillStyle = "#f2c9a8"; // legs
    ctx.fillRect(cx - W * 0.09, bodyTop + bodyH * 0.6, W * 0.06, bodyH * 0.4);
    ctx.fillRect(cx + W * 0.03, bodyTop + bodyH * 0.6, W * 0.06, bodyH * 0.4);
    ctx.fillStyle = "#ffffff"; // shoes
    ctx.fillRect(cx - W * 0.1, bodyTop + bodyH, W * 0.08, H * 0.03);
    ctx.fillRect(cx + W * 0.02, bodyTop + bodyH, W * 0.08, H * 0.03);
    stamp(ctx, W, H, `示範｜${label || DATA.styles[style] || ""}`);
    return c.toDataURL("image/png");
  }

  const ANALYSIS = {
    is_person: true,
    summary: "（示範內容）整體乾淨俐落，顏色協調；加一件有層次的外搭和更有質感的鞋子，會更有造型感。",
    current_outfit: {
      description: "白色圓領 T 恤搭配淺藍直筒牛仔褲、白色休閒鞋。",
      score: 7,
      strengths: ["配色簡單不出錯", "褲長剛好露出腳踝，比例俐落"],
      improvements: ["上衣略寬，紮一角或半紮可以拉出腰線", "加一件短版外套或背心增加層次"],
    },
    body_and_color: {
      body_notes: "肩線與臀線寬度相近，適合高腰下身與短版上衣，能拉長腿部比例。",
      flattering_colors: [{ name: "霧粉", hex: "#e8b4b8" }, { name: "燕麥", hex: "#d8c8b0" }, { name: "海軍藍", hex: "#1f2a44" }],
      colors_to_avoid: [{ name: "螢光黃", hex: "#e5ff3c" }],
    },
    suggestions: [
      { title: "法式週末", items: [{ category: "上衣", item: "細條紋船領針織衫", color: "白底藏青" }, { category: "下身", item: "高腰直筒牛仔褲", color: "淺藍" }, { category: "鞋子", item: "芭蕾平底鞋", color: "黑色" }], why: "條紋增加視覺重點，高腰拉長比例。" },
      { title: "溫柔通勤", items: [{ category: "外套", item: "短版西裝外套", color: "燕麥色" }, { category: "上衣", item: "絲光圓領上衣", color: "霧粉" }, { category: "下身", item: "高腰寬褲", color: "米白" }], why: "同色系層次顯得柔和又專業。" },
      { title: "街頭休閒", items: [{ category: "上衣", item: "短版連帽衛衣", color: "灰色" }, { category: "下身", item: "工裝寬褲", color: "卡其" }, { category: "鞋子", item: "復古球鞋", color: "白綠" }], why: "短版上衣搭寬褲，休閒又有比例。" },
    ],
  };

  const CHART = { category: "top", stretch: false, notes: "（示範尺寸表）版型偏寬鬆", sizes: [
    { size: "S", chest: 49, shoulder: 44, length: 66, sleeve: 20 },
    { size: "M", chest: 52, shoulder: 46, length: 68, sleeve: 21 },
    { size: "L", chest: 55, shoulder: 48, length: 70, sleeve: 22 },
    { size: "XL", chest: 58, shoulder: 50, length: 72, sleeve: 23 },
  ] };

  // Same comparison as outfit/sizing.py, kept short for the demo.
  function recommend(chart, reference) {
    const tight = new Set(["chest", "shoulder", "waist", "hip", "thigh"]);
    const keys = DATA.measures[chart.category];
    const ref = Object.fromEntries(Object.entries(reference || {}).filter(([k, v]) => k in keys && v > 0));
    if (!Object.keys(ref).length) return [{ error: "請先填寫至少一項你合身衣服的尺寸。" }, 400];
    const weight = { chest: 2, waist: 2, hip: 1.5, sleeve: 0.5 };
    const verdict = (k, d) => Math.abs(d) <= 1.5 ? "剛好" : (Math.abs(d) <= 4 ? "略" : "明顯") + (tight.has(k) ? (d < 0 ? "緊" : "寬鬆") : (d < 0 ? "短" : "長"));
    const sizes = chart.sizes.map((row) => {
      let total = 0, weights = 0;
      const fits = Object.keys(keys).filter((k) => k in ref && row[k] != null).map((k) => {
        const diff = Math.round((row[k] - ref[k]) * 10) / 10, w = weight[k] ?? 1;
        total += w * Math.abs(diff) * (diff < 0 && tight.has(k) ? (chart.stretch ? 0.75 : 1.5) : 1);
        weights += w;
        return { key: k, label: keys[k].label, garment: row[k], reference: ref[k], diff, verdict: verdict(k, diff) };
      });
      return { size: row.size, score: weights ? total / weights : Infinity, fits };
    }).filter((s) => s.fits.length);
    const best = sizes.reduce((a, b) => (b.score < a.score ? b : a), sizes[0]);
    return [{ recommended: best.size, sizes, warnings: [] }];
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (url, options = {}) => {
    const path = typeof url === "string" ? url : url.url;
    if (!path.startsWith("/api/")) return realFetch(url, options);
    const body = options.body ? JSON.parse(options.body) : {};
    switch (path) {
      case "/api/occasions": return json(DATA.occasions);
      case "/api/measures": return json(DATA.measures);
      case "/api/tryon-info": return json({ provider: "genlook", needs_garment_photo: true, max_garments: 4 });
      case "/api/avatar": return json({ styles: DATA.styles, templates: DATA.templates, avatar: savedAvatar });
      case "/api/avatar/template": {
        await wait(1500);
        const t = DATA.templates[body.template] || {};
        const colors = { city: "#c9b8a6", date: "#c2405a", office: "#3d4a63", resort: "#f2d58a", campus: "#7a8f5a", winter: "#8a6f5c" };
        return json({ image: demoAvatar("game", colors[body.template], null, `${t.icon || ""} ${t.label || ""}`) });
      }
      case "/api/analyze": await wait(1500); return json(ANALYSIS);
      case "/api/size-chart": await wait(1200); return json(CHART);
      case "/api/recommend-size": return json(...recommend(body.chart, body.reference));
      case "/api/try-on": await wait(2000); return json({ image: await demoTryOn(body.image, body.garments || []), note: "" });
      case "/api/avatar/preview": await wait(1200 + Math.random() * 1200); return json({ image: demoAvatar(body.style) });
      case "/api/avatar/save": savedAvatar = { image: body.image, style: body.style }; return json({ saved: true });
      case "/api/avatar/dress": {
        await wait(1500);
        const garment = body.garments && body.garments[0] ? await load(body.garments[0]) : null;
        return json({ image: demoAvatar(savedAvatar ? savedAvatar.style : "game", "#e8b4b8", garment) });
      }
      default: return json({ error: "示範版沒有這個功能" }, 404);
    }
  };

  window.KeepFace = {
    keepOriginalHead: async () => ({ reason: "示範版不做臉部處理；正式版會把你原本的臉貼回試穿照。" }),
  };
})();
