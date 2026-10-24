"""Send an outfit photo to Claude and get back a structured style analysis."""

from .claude import AnalysisError, call, obj
from .claude import build_request as _build_request

OCCASIONS = {"daily": "日常休閒", "work": "上班通勤", "date": "約會", "party": "聚會派對", "formal": "正式場合", "sport": "運動戶外"}

SYSTEM = """你是一位專業、友善的穿搭顧問。使用者會上傳一張自己的照片，請根據照片分析穿搭並給出建議。

原則：
- 全部用繁體中文回答，語氣正面、具體、實用。
- 談身形時只描述比例與線條（例如肩寬、腰線、腿長比例），用來說明哪些剪裁適合；不要評論胖瘦、外貌或給任何貶低的評語。
- 顏色建議依照片中可見的膚色、髮色判斷，並附上代表色的 hex 色碼。
- 建議要符合使用者指定的場合，單品要具體（例如「高腰直筒深藍牛仔褲」而不是「褲子」）。
- 如果照片裡看不到人或看不清楚穿著，把 is_person 設為 false，並在 summary 說明需要什麼樣的照片，其他欄位給空值即可。"""

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}


SCHEMA = obj(
    is_person={"type": "boolean"},
    summary=_STR,
    current_outfit=obj(description=_STR, score={"type": "integer"}, strengths=_STR_LIST, improvements=_STR_LIST),
    body_and_color=obj(
        body_notes=_STR,
        flattering_colors={"type": "array", "items": obj(name=_STR, hex=_STR)},
        colors_to_avoid={"type": "array", "items": obj(name=_STR, hex=_STR)},
    ),
    suggestions={
        "type": "array",
        "items": obj(
            title=_STR,
            items={"type": "array", "items": obj(category=_STR, item=_STR, color=_STR)},
            why=_STR,
        ),
    },
)


def build_request(image: bytes, media_type: str, occasion: str, note: str = "") -> dict:
    if occasion not in OCCASIONS:
        raise AnalysisError(f"未知的場合：{occasion}")
    prompt = (
        f"場合：{OCCASIONS[occasion]}\n"
        + (f"補充說明：{note.strip()}\n" if note.strip() else "")
        + "請分析照片中的穿搭：給目前穿搭 1–10 分、優點與可改進處，說明適合的剪裁與顏色，"
        "再提出 3 套適合這個場合的完整穿搭建議。"
    )
    return _build_request(SYSTEM, image, media_type, prompt, SCHEMA)


def analyze(image: bytes, media_type: str, occasion: str, note: str = "", client=None) -> dict:
    result = call(build_request(image, media_type, occasion, note), client)
    outfit = result["current_outfit"]
    outfit["score"] = max(1, min(10, outfit["score"]))
    return result
