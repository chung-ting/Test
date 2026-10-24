"""Size recommendation: read a size chart photo with Claude, then compare it with a garment that fits.

All measurements are flat garment measurements in cm (laid flat, one side), the way most
online size charts list them. The comparison itself is plain arithmetic so it's repeatable
and the user can check every number.
"""

from dataclasses import dataclass

from .claude import AnalysisError, build_request, call, obj


@dataclass(frozen=True)
class Measure:
    label: str
    hint: str
    weight: float
    girth: bool  # girths (chest, waist...) can be listed as circumferences; lengths can't


MEASURES = {
    "top": {
        "chest": Measure("胸寬", "兩邊腋下之間平量", 2.0, True),
        "shoulder": Measure("肩寬", "左右肩線接縫之間", 1.0, False),
        "length": Measure("衣長", "肩頸最高點量到下擺", 1.0, False),
        "sleeve": Measure("袖長", "肩線接縫量到袖口", 0.5, False),
    },
    "bottom": {
        "waist": Measure("腰寬", "扣好後腰頭左右平量", 2.0, True),
        "hip": Measure("臀寬", "褲襠上方最寬處平量", 1.5, True),
        "thigh": Measure("大腿寬", "褲襠正下方橫量單腿", 1.0, True),
        "length": Measure("褲長", "腰頭上緣沿外側量到褲口", 1.0, False),
    },
}
# Keys where a smaller garment means a tighter fit (as opposed to shorter).
TIGHT_KEYS = {"chest", "shoulder", "waist", "hip", "thigh"}
EXACT_CM = 1.5   # within this, a measurement counts as matching
SLIGHT_CM = 4.0  # within this, a slight difference; beyond, a clear one
TIGHT_PENALTY = 1.5  # being too tight is worse than too loose...
STRETCH_TIGHT_PENALTY = 0.75  # ...unless the fabric stretches

ALL_KEYS = sorted({k for m in MEASURES.values() for k in m})
_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}

CHART_SYSTEM = """你負責把服飾商品的尺寸表照片轉成結構化資料。

規則：
- 照抄表格裡的數字，不要自己推算或補上表格沒有的數值；沒有的欄位填 null。
- 數字是範圍（例如 88-92）時填中間值。
- unit：表格用公分填 cm，用英吋填 inch。
- girth_type：胸、腰、臀、大腿是「平量寬度」（例如胸寬 52）填 flat；是「一圈的圍度」（例如胸圍 104）填 circumference。
- chart_type：表格是衣服本身的尺寸填 garment；是建議的身體尺寸（例如「適合胸圍 88-92 的人」）填 body；看不出來填 unknown。
- category：上衣、外套、洋裝填 top；褲子、裙子填 bottom。
- 欄位對應：胸寬/胸圍→chest、肩寬→shoulder、衣長/褲長/裙長→length、袖長→sleeve、腰寬/腰圍→waist、臀寬/臀圍→hip、大腿寬/腿圍→thigh。
- stretch：表格或照片中有標示彈性布料時填 true。
- notes：用繁體中文簡短說明表格中其他重要資訊（例如版型偏大、寬鬆版），沒有就填空字串。"""

CHART_SCHEMA = obj(
    is_size_chart={"type": "boolean"},
    chart_type={"type": "string", "enum": ["garment", "body", "unknown"]},
    category={"type": "string", "enum": ["top", "bottom"]},
    unit={"type": "string", "enum": ["cm", "inch"]},
    girth_type={"type": "string", "enum": ["flat", "circumference"]},
    stretch={"type": "boolean"},
    notes={"type": "string"},
    sizes={"type": "array", "items": obj(size={"type": "string"}, **{k: _NUM for k in ALL_KEYS})},
)


def normalize_chart(raw: dict) -> dict:
    """Convert Claude's reading of a chart into flat cm measurements for the chart's category."""
    if not raw["is_size_chart"] or not raw["sizes"]:
        raise AnalysisError("看不出這是尺寸表，請截一張清楚的尺寸表。")
    if raw["chart_type"] == "body":
        raise AnalysisError("這是「身體尺寸」建議表，不是衣服本身的尺寸。請找商品頁上的平量尺寸表（例如胸寬、衣長）。")
    scale = 2.54 if raw["unit"] == "inch" else 1.0
    measures = MEASURES[raw["category"]]
    sizes = []
    for row in raw["sizes"]:
        values = {}
        for key, m in measures.items():
            v = row.get(key)
            if v is None:
                continue
            v *= scale
            if m.girth and raw["girth_type"] == "circumference":
                v /= 2
            values[key] = round(v, 1)
        if values:
            sizes.append({"size": row["size"], **values})
    if not sizes:
        raise AnalysisError("尺寸表裡沒有可以比對的尺寸欄位。")
    return {"category": raw["category"], "stretch": raw["stretch"], "notes": raw["notes"], "sizes": sizes}


def extract_chart(image: bytes, media_type: str, client=None) -> dict:
    request = build_request(CHART_SYSTEM, image, media_type, "請讀取這張尺寸表。", CHART_SCHEMA)
    return normalize_chart(call(request, client))


def _verdict(key: str, diff: float) -> str:
    size = abs(diff)
    if size <= EXACT_CM:
        return "剛好"
    degree = "略" if size <= SLIGHT_CM else "明顯"
    if key in TIGHT_KEYS:
        return degree + ("緊" if diff < 0 else "寬鬆")
    return degree + ("短" if diff < 0 else "長")


def _number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 500:
        raise AnalysisError(f"{what} 的數值不正確")
    return float(value)


def recommend(chart: dict, reference: dict) -> dict:
    """Rank the chart's sizes by how close each is to the reference garment.

    `chart` is the output of normalize_chart (possibly edited by the user);
    `reference` maps measurement keys to the user's well-fitting garment, in flat cm.
    """
    category = chart.get("category")
    if category not in MEASURES:
        raise AnalysisError("尺寸表的類別不正確")
    measures = MEASURES[category]
    ref = {k: _number(v, measures[k].label) for k, v in reference.items() if k in measures and v not in (None, "")}
    if not ref:
        raise AnalysisError("請先填寫至少一項你合身衣服的尺寸。")
    tight_penalty = STRETCH_TIGHT_PENALTY if chart.get("stretch") else TIGHT_PENALTY

    ranked = []
    for row in chart.get("sizes", []):
        fits, total, weights = [], 0.0, 0.0
        for key, m in measures.items():
            if key not in ref or row.get(key) is None:
                continue
            garment = _number(row[key], f"{row.get('size')} 的{m.label}")
            diff = round(garment - ref[key], 1)
            penalty = abs(diff) * (tight_penalty if diff < 0 and key in TIGHT_KEYS else 1.0)
            total += m.weight * penalty
            weights += m.weight
            fits.append({"key": key, "label": m.label, "garment": garment, "reference": ref[key],
                         "diff": diff, "verdict": _verdict(key, diff)})
        if fits:
            ranked.append({"size": str(row.get("size", "")), "score": round(total / weights, 2), "fits": fits})
    if not ranked:
        raise AnalysisError("尺寸表和你填的尺寸沒有共同的項目可以比對。")

    best = min(ranked, key=lambda r: r["score"])
    warnings = []
    girths = [f for f in best["fits"] if measures[f["key"]].girth]
    if not girths:
        warnings.append("沒有比對到胸寬或腰寬這類圍度，推薦只依長度判斷，準確度較低。")
    if any(f["diff"] < -SLIGHT_CM for f in girths):
        warnings.append("最接近的尺寸仍然明顯偏緊，這件可能沒有適合的尺寸。")
    if any(f["diff"] > SLIGHT_CM for f in girths):
        warnings.append("最接近的尺寸仍然明顯偏寬鬆，如果不是想穿寬鬆版，這件可能沒有適合的尺寸。")
    return {"recommended": best["size"], "sizes": ranked, "warnings": warnings}
