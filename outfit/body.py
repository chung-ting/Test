"""Estimate body measurements from a front and a side photo plus the person's height.

The browser runs MediaPipe pose detection and person segmentation on each photo and sends
only derived data (no photo): pose landmarks in pixels and, for each pixel row, the
horizontal runs [left, right] where the person mask is set. From that:

- scale: the silhouette spans head-to-feet, so cm per pixel = height / silhouette height;
- widths come from the front silhouette, depths from the side one, at matching torso levels
  (located relative to the shoulder and hip landmarks of each photo);
- circumferences treat each cross-section as an ellipse, then apply a per-person
  calibration factor learned from one tape measurement.
"""

import math

from .claude import AnalysisError

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
PAIRS = ((L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP))
MIN_VISIBILITY = 0.5

# Torso levels as a fraction of the way from the shoulder line (0) to the hip-joint line (1).
CHEST_LEVEL = 0.28
WAIST_BAND = (0.45, 0.8)  # natural waist: narrowest point in this band
HIP_BAND = (0.85, 1.15)   # hip: widest point in this band

LABELS = {"shoulder": "肩寬", "chest": "胸圍", "waist": "腰圍", "hip": "臀圍", "inseam": "內側腿長", "arm": "手臂長"}
CALIBRATABLE = ("chest", "waist", "hip")


def ellipse_circumference(width: float, depth: float) -> float:
    """Ramanujan's approximation for an ellipse with the given full width and depth."""
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


class Silhouette:
    """One photo's pose landmarks and person-mask runs, validated."""

    def __init__(self, data: dict, name: str, profile: bool = False):
        self.name = name
        try:
            self.height = int(data["height"])
            self.landmarks = [(float(x), float(y), float(v)) for x, y, v in data["landmarks"]]
            self.rows = [[(int(l), int(r)) for l, r in row] for row in data["rows"]]
        except (KeyError, TypeError, ValueError):
            raise AnalysisError(f"{name}的資料格式錯誤，請重新選照片") from None
        if len(self.landmarks) != 33 or len(self.rows) != self.height:
            raise AnalysisError(f"{name}的資料格式錯誤，請重新選照片")
        # From the side the far shoulder and hip are hidden, so one of each pair is enough.
        seen = any if profile else all
        if not all(seen(self.landmarks[i][2] >= MIN_VISIBILITY for i in pair) for pair in PAIRS):
            raise AnalysisError(f"{name}看不清楚肩膀或臀部，請拍完整的全身照")
        filled = [y for y, row in enumerate(self.rows) if row]
        if not filled:
            raise AnalysisError(f"{name}偵測不到人")
        self.top, self.bottom = filled[0], filled[-1]
        if self.top <= 1 or self.bottom >= self.height - 2:
            raise AnalysisError(f"{name}的頭頂或腳底被切到了，請讓全身完整入鏡，上下留一點空間")

    def y(self, *indices: int) -> float:
        return sum(self.landmarks[i][1] for i in indices) / len(indices)

    def x(self, *indices: int) -> float:
        return sum(self.landmarks[i][0] for i in indices) / len(indices)

    @property
    def pixel_height(self) -> int:
        return self.bottom - self.top

    def level_y(self, fraction: float) -> int:
        shoulder, hip = self.y(L_SHOULDER, R_SHOULDER), self.y(L_HIP, R_HIP)
        return round(shoulder + fraction * (hip - shoulder))

    def torso_width(self, y: int) -> int:
        """Width of the mask run containing the body's center line at row y (ignores separated arms)."""
        cx = self.x(L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        runs = self.rows[y] if 0 <= y < self.height else []
        if not runs:
            return 0
        l, r = min(runs, key=lambda run: 0 if run[0] <= cx <= run[1] else min(abs(run[0] - cx), abs(run[1] - cx)))
        return r - l + 1

    def extreme_in_band(self, band: tuple[float, float], widest: bool, max_y: int | None = None) -> tuple[int, int]:
        """(y, width) of the narrowest or widest torso row between two torso levels, not below max_y."""
        end = self.level_y(band[1]) if max_y is None else min(self.level_y(band[1]), max_y)
        ys = range(self.level_y(band[0]), end + 1)
        rows = [(y, self.torso_width(y)) for y in ys]
        rows = [r for r in rows if r[1] > 0]
        if not rows:
            raise AnalysisError(f"{self.name}的軀幹輪廓不完整，請換一張背景單純的照片")
        return (max if widest else min)(rows, key=lambda r: r[1])

    def crotch_y(self) -> int | None:
        """First row below the hips where the legs separate (two runs straddle the center line)."""
        cx = self.x(L_HIP, R_HIP)
        for y in range(round(self.y(L_HIP, R_HIP)), self.bottom):
            runs = self.rows[y]
            if any(r < cx for _, r in runs) and any(l > cx for l, _ in runs) and not any(l <= cx <= r for l, r in runs):
                return y
        return None

    def distance(self, a: int, b: int) -> float:
        (xa, ya, _), (xb, yb, _) = self.landmarks[a], self.landmarks[b]
        return math.hypot(xa - xb, ya - yb)


def measure(front: dict, side: dict, height_cm: float, calibration: dict | None = None) -> dict:
    if isinstance(height_cm, bool) or not isinstance(height_cm, (int, float)) or not 100 <= height_cm <= 230:
        raise AnalysisError("請輸入正確的身高（100–230 公分）")
    f, s = Silhouette(front, "正面照"), Silhouette(side, "側面照", profile=True)
    f_scale, s_scale = height_cm / f.pixel_height, height_cm / s.pixel_height

    # Shoulders look wide from the front and nearly overlap from the side.
    f_ratio = f.distance(L_SHOULDER, R_SHOULDER) / f.pixel_height
    s_ratio = s.distance(L_SHOULDER, R_SHOULDER) / s.pixel_height
    if f_ratio < 0.12:
        raise AnalysisError("正面照看起來不是正面，請面向鏡頭站直")
    if s_ratio > 0.1:
        raise AnalysisError("側面照看起來不是側面，請轉 90 度、身體側面對著鏡頭")

    chest_f, chest_s = f.level_y(CHEST_LEVEL), s.level_y(CHEST_LEVEL)
    crotch = f.crotch_y()
    waist_f, waist_w = f.extreme_in_band(WAIST_BAND, widest=False)
    # Stop above the crotch so spread thighs aren't mistaken for hips.
    hip_f, hip_w = f.extreme_in_band(HIP_BAND, widest=True, max_y=None if crotch is None else crotch - 3)
    waist_s, waist_d = s.extreme_in_band(WAIST_BAND, widest=False)
    hip_s, hip_d = s.extreme_in_band(HIP_BAND, widest=True)
    raw = {
        "chest": ellipse_circumference(f.torso_width(chest_f) * f_scale, s.torso_width(chest_s) * s_scale),
        "waist": ellipse_circumference(waist_w * f_scale, waist_d * s_scale),
        "hip": ellipse_circumference(hip_w * f_scale, hip_d * s_scale),
        # Landmarks sit on the joint centers, a little inside the shoulder points.
        "shoulder": f.distance(L_SHOULDER, R_SHOULDER) * f_scale * 1.1,
        "arm": (f.distance(L_SHOULDER, L_ELBOW) + f.distance(L_ELBOW, L_WRIST)
                + f.distance(R_SHOULDER, R_ELBOW) + f.distance(R_ELBOW, R_WRIST)) / 2 * f_scale,
    }
    warnings = []
    if crotch is None:
        warnings.append("正面照看不到雙腿之間的空隙，量不到內側腿長。拍照時雙腳請稍微打開。")
    else:
        raw["inseam"] = (f.bottom - crotch) * f_scale
    if any(f.landmarks[i][2] < MIN_VISIBILITY for i in (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)):
        del raw["arm"]
        warnings.append("正面照看不清楚手臂，量不到手臂長。")

    factors = {}
    for key, value in (calibration or {}).items():
        if key in CALIBRATABLE and isinstance(value, (int, float)) and not isinstance(value, bool) and 0.8 <= value <= 1.25:
            factors[key] = float(value)
    measurements = [
        {"key": k, "label": LABELS[k], "raw": round(v, 1), "value": round(v * factors.get(k, 1.0), 1),
         "calibrated": k in factors}
        for k, v in raw.items()
    ]
    if raw["waist"] > raw["chest"] * 1.15 or raw["waist"] > raw["hip"] * 1.15:
        warnings.append("腰圍算出來比胸圍或臀圍大很多，可能是衣服太寬鬆或手擋到腰，建議穿貼身衣服重拍。")
    if not 0.75 <= raw["hip"] / raw["chest"] <= 1.4:
        warnings.append("胸圍和臀圍的比例不太合理，可能是姿勢不標準（例如雙腳張太開）或手擋到身體，建議照說明重拍。")
    return {
        "measurements": sorted(measurements, key=lambda m: list(LABELS).index(m["key"])),
        "warnings": warnings,
        # Rows where each measurement was taken, so the page can draw them on the photos.
        "levels": {
            "front": {"chest": chest_f, "waist": waist_f, "hip": hip_f, **({"crotch": crotch} if crotch else {})},
            "side": {"chest": chest_s, "waist": waist_s, "hip": hip_s},
        },
    }
