import math
import unittest

from outfit import AnalysisError
from outfit.body import ellipse_circumference, measure

# A synthetic 800 px tall person (rows 100..900 of a 1000 px image) standing for 170 cm.
HEIGHT_CM = 170
SCALE = HEIGHT_CM / 800
CX = 500
SHOULDER_Y, HIP_Y, CROTCH_Y = 250, 480, 540


def torso(y, chest, waist, hip):
    """Piecewise-linear torso width: chest at level 0.28, narrowest at 0.62, widest at the hip joint."""
    level = (y - SHOULDER_Y) / (HIP_Y - SHOULDER_Y)
    points = [(-0.2, chest * 0.9), (0.28, chest), (0.62, waist), (1.0, hip), (1.3, hip * 0.9)]
    for (l0, w0), (l1, w1) in zip(points, points[1:]):
        if l0 <= level <= l1:
            return w0 + (w1 - w0) * (level - l0) / (l1 - l0)
    raise ValueError(level)


def silhouette(chest, waist, hip, shoulder_dx, arms=True, legs_apart=True):
    rows = []
    for y in range(1000):
        runs = []
        if 100 <= y < 220:  # head and neck
            runs.append((CX - 40, CX + 40))
        elif 220 <= y < CROTCH_Y:
            w = torso(y, chest, waist, hip)
            runs.append((round(CX - w / 2), round(CX + w / 2) - 1))
            if arms and 260 <= y < 520:  # arms held slightly away from the body
                runs = [(CX - 200, CX - 170)] + runs + [(CX + 170, CX + 200)]
        elif CROTCH_Y <= y <= 900:
            runs = [(CX - 70, CX - 5), (CX + 5, CX + 70)] if legs_apart else [(CX - 70, CX + 70)]
        rows.append(runs)
    lm = [(CX, 150, 0.99)] * 33
    lm[11], lm[12] = (CX + shoulder_dx, SHOULDER_Y, 0.99), (CX - shoulder_dx, SHOULDER_Y, 0.99)
    lm[13], lm[14] = (CX + 185, 380, 0.99), (CX - 185, 380, 0.99)
    lm[15], lm[16] = (CX + 190, 510, 0.99), (CX - 190, 510, 0.99)
    lm[23], lm[24] = (CX + 60, HIP_Y, 0.99), (CX - 60, HIP_Y, 0.99)
    return {"height": 1000, "landmarks": [list(p) for p in lm], "rows": [[list(r) for r in row] for row in rows]}


FRONT = silhouette(chest=150, waist=120, hip=160, shoulder_dx=90)
SIDE = silhouette(chest=110, waist=90, hip=110, shoulder_dx=5, arms=False, legs_apart=False)


def by_key(result):
    return {m["key"]: m for m in result["measurements"]}


class BodyTests(unittest.TestCase):
    def test_ellipse_circumference(self):
        self.assertAlmostEqual(ellipse_circumference(10, 10), math.pi * 10)
        self.assertAlmostEqual(ellipse_circumference(20, 10), 48.44, places=2)

    def test_recovers_known_measurements(self):
        result = measure(FRONT, SIDE, HEIGHT_CM)
        m = by_key(result)
        expected = {
            "chest": ellipse_circumference(150 * SCALE, 110 * SCALE),
            "waist": ellipse_circumference(120 * SCALE, 90 * SCALE),
            "hip": ellipse_circumference(160 * SCALE, 110 * SCALE),
            "shoulder": 180 * SCALE * 1.1,
            "inseam": (900 - CROTCH_Y) * SCALE,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(m[key]["value"], value, delta=1.0, msg=key)
        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["levels"]["front"]["crotch"], CROTCH_Y)

    def test_calibration_scales_girths_only(self):
        m = by_key(measure(FRONT, SIDE, HEIGHT_CM, {"waist": 1.1, "inseam": 1.2, "chest": 3}))
        self.assertAlmostEqual(m["waist"]["value"], m["waist"]["raw"] * 1.1, delta=0.1)
        self.assertTrue(m["waist"]["calibrated"])
        self.assertFalse(m["chest"]["calibrated"])  # out-of-range factor ignored
        self.assertEqual(m["inseam"]["value"], m["inseam"]["raw"])

    def test_missing_leg_gap_warns_instead_of_failing(self):
        closed = silhouette(chest=150, waist=120, hip=160, shoulder_dx=90, legs_apart=False)
        result = measure(closed, SIDE, HEIGHT_CM)
        self.assertNotIn("inseam", by_key(result))
        self.assertIn("內側腿長", result["warnings"][0])

    def test_hip_stops_above_spread_legs(self):
        # Legs separate at row 500 (inside the hip search band) and splay wider than the hips.
        wide = silhouette(chest=150, waist=120, hip=160, shoulder_dx=90)
        for y in range(500, CROTCH_Y):
            wide["rows"][y] = [[CX - 200, CX - 5], [CX + 5, CX + 200]]
        result = measure(wide, SIDE, HEIGHT_CM)
        self.assertEqual(result["levels"]["front"]["crotch"], 500)
        self.assertAlmostEqual(by_key(result)["hip"]["value"], by_key(measure(FRONT, SIDE, HEIGHT_CM))["hip"]["value"])

    def test_warns_on_implausible_proportions(self):
        big_hips = silhouette(chest=100, waist=90, hip=230, shoulder_dx=90)
        self.assertIn("比例", " ".join(measure(big_hips, SIDE, HEIGHT_CM)["warnings"]))

    def test_rejects_bad_photos(self):
        cases = {
            "看起來不是正面": (SIDE, SIDE),
            "看起來不是側面": (FRONT, FRONT),
        }
        for message, (front, side) in cases.items():
            with self.assertRaisesRegex(AnalysisError, message):
                measure(front, side, HEIGHT_CM)
        cut = dict(FRONT, rows=[[[480, 520]]] * 1000)
        with self.assertRaisesRegex(AnalysisError, "被切到"):
            measure(cut, SIDE, HEIGHT_CM)
        hidden = dict(FRONT, landmarks=[[x, y, 0.1 if i == 11 else v] for i, (x, y, v) in enumerate(FRONT["landmarks"])])
        with self.assertRaisesRegex(AnalysisError, "肩膀或臀部"):
            measure(hidden, SIDE, HEIGHT_CM)
        for height in (50, "170", True):
            with self.assertRaisesRegex(AnalysisError, "身高"):
                measure(FRONT, SIDE, height)
        with self.assertRaisesRegex(AnalysisError, "格式錯誤"):
            measure({"height": 1000}, SIDE, HEIGHT_CM)


if __name__ == "__main__":
    unittest.main()
