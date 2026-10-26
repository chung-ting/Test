// Browser port of outfit/body.py so the standalone page needs no server.
// Keep the two in sync; tests/test_body.py checks they agree on the same inputs.
(function (root) {
  "use strict";

  const L_SHOULDER = 11, R_SHOULDER = 12, L_ELBOW = 13, R_ELBOW = 14, L_WRIST = 15, R_WRIST = 16;
  const L_HIP = 23, R_HIP = 24;
  const PAIRS = [[L_SHOULDER, R_SHOULDER], [L_HIP, R_HIP]];
  const MIN_VISIBILITY = 0.5;

  const CHEST_LEVEL = 0.28;
  const WAIST_BAND = [0.45, 0.8];
  const HIP_BAND = [0.85, 1.15];

  const LABELS = { shoulder: "肩寬", chest: "胸圍", waist: "腰圍", hip: "臀圍", inseam: "內側腿長", arm: "手臂長" };
  const CALIBRATABLE = ["chest", "waist", "hip"];

  class MeasureError extends Error {}

  function ellipseCircumference(width, depth) {
    const a = width / 2, b = depth / 2;
    return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
  }

  // Python's round(): halves go to the even neighbour.
  function pyRound(v) {
    const r = Math.round(v);
    return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
  }
  const round1 = (v) => Math.round(v * 10) / 10;

  class Silhouette {
    constructor(data, name, profile = false) {
      this.name = name;
      if (!data || !Array.isArray(data.landmarks) || !Array.isArray(data.rows) ||
          data.landmarks.length !== 33 || data.rows.length !== data.height) {
        throw new MeasureError(`${name}的資料格式錯誤，請重新選照片`);
      }
      this.height = data.height;
      this.landmarks = data.landmarks;
      this.rows = data.rows;
      // From the side the far shoulder and hip are hidden, so one of each pair is enough.
      const seen = (pair) => (profile ? pair.some : pair.every).call(pair, (i) => this.landmarks[i][2] >= MIN_VISIBILITY);
      if (!PAIRS.every(seen)) throw new MeasureError(`${name}看不清楚肩膀或臀部，請拍完整的全身照`);
      const filled = [];
      this.rows.forEach((row, y) => { if (row.length) filled.push(y); });
      if (!filled.length) throw new MeasureError(`${name}偵測不到人`);
      this.top = filled[0];
      this.bottom = filled[filled.length - 1];
      if (this.top <= 1 || this.bottom >= this.height - 2) {
        throw new MeasureError(`${name}的頭頂或腳底被切到了，請讓全身完整入鏡，上下留一點空間`);
      }
    }

    y(...ids) { return ids.reduce((s, i) => s + this.landmarks[i][1], 0) / ids.length; }
    x(...ids) { return ids.reduce((s, i) => s + this.landmarks[i][0], 0) / ids.length; }
    get pixelHeight() { return this.bottom - this.top; }

    levelY(fraction) {
      const shoulder = this.y(L_SHOULDER, R_SHOULDER), hip = this.y(L_HIP, R_HIP);
      return pyRound(shoulder + fraction * (hip - shoulder));
    }

    torsoWidth(y) {
      const cx = this.x(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP);
      const runs = y >= 0 && y < this.height ? this.rows[y] : [];
      if (!runs.length) return 0;
      const dist = ([l, r]) => (l <= cx && cx <= r ? 0 : Math.min(Math.abs(l - cx), Math.abs(r - cx)));
      let best = runs[0];
      for (const run of runs) if (dist(run) < dist(best)) best = run;
      return best[1] - best[0] + 1;
    }

    extremeInBand(band, widest, maxY = null) {
      const end = maxY === null ? this.levelY(band[1]) : Math.min(this.levelY(band[1]), maxY);
      let best = null;
      for (let y = this.levelY(band[0]); y <= end; y++) {
        const w = this.torsoWidth(y);
        if (w > 0 && (!best || (widest ? w > best[1] : w < best[1]))) best = [y, w];
      }
      if (!best) throw new MeasureError(`${this.name}的軀幹輪廓不完整，請換一張背景單純的照片`);
      return best;
    }

    crotchY() {
      const cx = this.x(L_HIP, R_HIP);
      for (let y = pyRound(this.y(L_HIP, R_HIP)); y < this.bottom; y++) {
        const runs = this.rows[y];
        if (runs.some(([, r]) => r < cx) && runs.some(([l]) => l > cx) && !runs.some(([l, r]) => l <= cx && cx <= r)) return y;
      }
      return null;
    }

    distance(a, b) {
      const [xa, ya] = this.landmarks[a], [xb, yb] = this.landmarks[b];
      return Math.hypot(xa - xb, ya - yb);
    }
  }

  function measure(front, side, heightCm, calibration = {}) {
    if (typeof heightCm !== "number" || !(heightCm >= 100 && heightCm <= 230)) {
      throw new MeasureError("請輸入正確的身高（100–230 公分）");
    }
    const f = new Silhouette(front, "正面照"), s = new Silhouette(side, "側面照", true);
    const fScale = heightCm / f.pixelHeight, sScale = heightCm / s.pixelHeight;

    // Shoulders look wide from the front and nearly overlap from the side.
    if (f.distance(L_SHOULDER, R_SHOULDER) / f.pixelHeight < 0.12) {
      throw new MeasureError("正面照看起來不是正面，請面向鏡頭站直");
    }
    if (s.distance(L_SHOULDER, R_SHOULDER) / s.pixelHeight > 0.1) {
      throw new MeasureError("側面照看起來不是側面，請轉 90 度、身體側面對著鏡頭");
    }

    const chestF = f.levelY(CHEST_LEVEL), chestS = s.levelY(CHEST_LEVEL);
    const crotch = f.crotchY();
    const [waistF, waistW] = f.extremeInBand(WAIST_BAND, false);
    // Stop above the crotch so spread thighs aren't mistaken for hips.
    const [hipF, hipW] = f.extremeInBand(HIP_BAND, true, crotch === null ? null : crotch - 3);
    const [waistS, waistD] = s.extremeInBand(WAIST_BAND, false);
    const [hipS, hipD] = s.extremeInBand(HIP_BAND, true);
    const raw = {
      chest: ellipseCircumference(f.torsoWidth(chestF) * fScale, s.torsoWidth(chestS) * sScale),
      waist: ellipseCircumference(waistW * fScale, waistD * sScale),
      hip: ellipseCircumference(hipW * fScale, hipD * sScale),
      // Landmarks sit on the joint centers, a little inside the shoulder points.
      shoulder: f.distance(L_SHOULDER, R_SHOULDER) * fScale * 1.1,
      arm: (f.distance(L_SHOULDER, L_ELBOW) + f.distance(L_ELBOW, L_WRIST)
            + f.distance(R_SHOULDER, R_ELBOW) + f.distance(R_ELBOW, R_WRIST)) / 2 * fScale,
    };
    const warnings = [];
    if (crotch === null) {
      warnings.push("正面照看不到雙腿之間的空隙，量不到內側腿長。拍照時雙腳請稍微打開。");
    } else {
      raw.inseam = (f.bottom - crotch) * fScale;
    }
    if ([L_ELBOW, R_ELBOW, L_WRIST, R_WRIST].some((i) => f.landmarks[i][2] < MIN_VISIBILITY)) {
      delete raw.arm;
      warnings.push("正面照看不清楚手臂，量不到手臂長。");
    }

    const factors = {};
    for (const [key, value] of Object.entries(calibration || {})) {
      if (CALIBRATABLE.includes(key) && typeof value === "number" && value >= 0.8 && value <= 1.25) factors[key] = value;
    }
    const measurements = Object.keys(LABELS).filter((k) => k in raw).map((k) => ({
      key: k, label: LABELS[k], raw: round1(raw[k]), value: round1(raw[k] * (factors[k] ?? 1)), calibrated: k in factors,
    }));
    if (raw.waist > raw.chest * 1.15 || raw.waist > raw.hip * 1.15) {
      warnings.push("腰圍算出來比胸圍或臀圍大很多，可能是衣服太寬鬆或手擋到腰，建議穿貼身衣服重拍。");
    }
    const ratio = raw.hip / raw.chest;
    if (!(ratio >= 0.75 && ratio <= 1.4)) {
      warnings.push("胸圍和臀圍的比例不太合理，可能是姿勢不標準（例如雙腳張太開）或手擋到身體，建議照說明重拍。");
    }
    return {
      measurements,
      warnings,
      levels: {
        front: { chest: chestF, waist: waistF, hip: hipF, ...(crotch ? { crotch } : {}) },
        side: { chest: chestS, waist: waistS, hip: hipS },
      },
    };
  }

  const api = { measure, ellipseCircumference, MeasureError, LABELS, CALIBRATABLE };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BodyMeasure = api;
})(typeof self !== "undefined" ? self : this);
