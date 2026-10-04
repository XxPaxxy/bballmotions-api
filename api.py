import os, math, tempfile
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

REF_DIR = "output_v3"
POSE_MODEL = "pose_landmarker_lite.task"
HAND_MODEL = "hand_landmarker.task"
USE_HANDS = os.path.exists(HAND_MODEL)
SHOOTING_ARM = "r"
N_POINTS = 100
MIN_VIS = 0.5
PHASES = ["setup and dip", "release", "follow-through"]

SIDES = {
    "l": dict(shoulder=11, elbow=13, wrist=15, index=19, hip=23, knee=25, ankle=27),
    "r": dict(shoulder=12, elbow=14, wrist=16, index=20, hip=24, knee=26, ankle=28),
}
GUIDE_ARM = "l" if SHOOTING_ARM == "r" else "r"
ARMS = {"shoot": SIDES[SHOOTING_ARM], "guide": SIDES[GUIDE_ARM]}

NAMES = {
    "shoot_elbow_angle": "shooting elbow", "shoot_wrist_angle": "shooting wrist",
    "shoot_shoulder_angle": "shooting arm raise", "guide_elbow_angle": "guide elbow",
    "guide_wrist_angle": "guide wrist", "guide_shoulder_angle": "guide arm raise",
    "hip_height": "hip dip and drive", "torso_lean": "torso lean",
    "nose_x_offset": "head side position", "nose_y_offset": "head height",
    "finger_spread": "finger spread", "index_extension": "index finger extension",
    "shoot_knee_angle": "knee bend", "guide_knee_angle": "other knee bend",
    "shoot_thigh_angle": "thigh angle", "guide_thigh_angle": "other thigh angle",
}


def angle(a, b, c):
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def process_video(path):
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    hand_ctx = None
    if USE_HANDS:
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO, num_hands=2)
        hand_ctx = vision.HandLandmarker.create_from_options(hand_opts)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows = []
    with vision.PoseLandmarker.create_from_options(pose_opts) as pose_lm:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(i * 1000 / fps)
            res = pose_lm.detect_for_video(mp_img, ts)
            row = {}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                P = lambda idx: np.array([lms[idx].x * w, lms[idx].y * h])
                vis = lambda idx: (lms[idx].visibility or 0)
                sh_mid = (P(11) + P(12)) / 2
                hip_mid = (P(23) + P(24)) / 2
                torso = np.linalg.norm(sh_mid - hip_mid) + 1e-9
                for arm, ix in ARMS.items():
                    s, e, wr, idx, hp = (P(ix[k]) for k in ("shoulder", "elbow", "wrist", "index", "hip"))
                    row[f"{arm}_elbow_angle"] = angle(s, e, wr)
                    row[f"{arm}_wrist_angle"] = angle(e, wr, idx)
                    row[f"{arm}_shoulder_angle"] = angle(hp, s, e)
                    if vis(ix["hip"]) > MIN_VIS and vis(ix["knee"]) > MIN_VIS:
                        v = P(ix["knee"]) - P(ix["hip"])
                        row[f"{arm}_thigh_angle"] = math.degrees(math.atan2(abs(v[0]), v[1]))
                        if vis(ix["ankle"]) > MIN_VIS:
                            row[f"{arm}_knee_angle"] = angle(P(ix["hip"]), P(ix["knee"]), P(ix["ankle"]))
                row["hip_y_raw"] = hip_mid[1] / torso
                v = sh_mid - hip_mid
                row["torso_lean"] = math.degrees(math.atan2(v[0], -v[1]))
                nose = P(0)
                row["nose_x_offset"] = (nose[0] - sh_mid[0]) / torso
                row["nose_y_offset"] = (sh_mid[1] - nose[1]) / torso
                if hand_ctx is not None:
                    hres = hand_ctx.detect_for_video(mp_img, ts)
                    if hres.hand_landmarks:
                        pose_wrist = P(ARMS["shoot"]["wrist"])
                        best, best_d = None, 1e9
                        for hl in hres.hand_landmarks:
                            d = np.linalg.norm(np.array([hl[0].x * w, hl[0].y * h]) - pose_wrist)
                            if d < best_d:
                                best, best_d = hl, d
                        if best is not None and best_d < torso * 0.6:
                            H = lambda k: np.array([best[k].x * w, best[k].y * h])
                            row["finger_spread"] = angle(H(8), H(0), H(20))
                            palm = np.linalg.norm(H(5) - H(0)) + 1e-9
                            row["index_extension"] = np.linalg.norm(H(8) - H(0)) / palm
            rows.append(row)
            i += 1
    cap.release()
    if hand_ctx is not None:
        hand_ctx.close()
    df = pd.DataFrame(rows)
    if "hip_y_raw" in df and df["hip_y_raw"].notna().any():
        df["hip_height"] = -(df["hip_y_raw"] - df["hip_y_raw"].dropna().iloc[0])
    return df


def normalize(df, cols):
    out = {}
    for c in cols:
        if c not in df or df[c].notna().sum() < 10:
            out[c] = np.full(N_POINTS, np.nan)
            continue
        s = df[c].interpolate(limit_direction="both").rolling(5, center=True, min_periods=1).mean()
        out[c] = np.interp(np.linspace(0, len(s) - 1, N_POINTS), np.arange(len(s)), s)
    return pd.DataFrame(out)


# ---- load the reference once, when the server starts ----
AVG = pd.read_csv(os.path.join(REF_DIR, "average_shot.csv"), index_col="pct_of_shot")
_shots = pd.read_csv(os.path.join(REF_DIR, "all_shots_normalized.csv"))
STD = _shots.groupby("pct_of_shot")[list(AVG.columns)].std()


def score_clip(path):
    new = normalize(process_video(path), list(AVG.columns))
    scores = {}
    for c in AVG.columns:
        if not (AVG[c].notna().all() and STD[c].notna().all()) or new[c].isna().any():
            continue
        sigma = np.maximum(STD[c].values, 0.25 * np.nanmean(STD[c].values) + 1e-6)
        z = np.abs(new[c].values - AVG[c].values) / sigma
        scores[c] = float(np.mean(np.exp(-0.5 * (z / 2) ** 2)) * 100)
    return new, scores


def make_feedback(new, scores, overall):
    items = []
    for c, s in scores.items():
        d = new[c].values - AVG[c].values
        phases = [float(np.mean(d[a:b])) for a, b in ((0, 33), (33, 66), (66, 100))]
        items.append((s, c, phases))
    items.sort()
    if overall >= 80:
        opener = "Strong form. You're very close to the top shooters, so these are small refinements."
    elif overall >= 60:
        opener = "Solid base. A few clear differences from the top shooters are holding your score back."
    else:
        opener = "There's real room to improve. Fix these one at a time instead of all at once."
    tips = []
    for s, c, ph in items[:3]:
        wp = int(np.argmax(np.abs(ph)))
        direction = "higher" if ph[wp] > 0 else "lower"
        tips.append({
            "marking": NAMES.get(c, c.replace("_", " ")),
            "score": round(s),
            "text": f"Your {NAMES.get(c, c)} differs most during the {PHASES[wp]}: "
                    f"it reads {direction} than the top shooters' average.",
        })
    best = items[-1]
    strength = f"Your {NAMES.get(best[1], best[1])} matches the top shooters well."
    return opener, tips, strength


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bballmotions.tech", "https://www.bballmotions.tech",
                   "http://localhost:5173", "http://127.0.0.1:5500"],
    allow_origin_regex=r"https://.*\.netlify\.app",
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
def analyze(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file.file.read())
    tmp.close()
    try:
        new, scores = score_clip(tmp.name)
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=422,
            detail="Video analysis failed. Check the server logs."
        )
    finally:
        os.unlink(tmp.name)
    if not scores:
        raise HTTPException(status_code=422,
                            detail="Couldn't find the shooter. Keep the upper body clearly in frame.")
    overall = float(np.mean(list(scores.values())))
    opener, tips, strength = make_feedback(new, scores, overall)
    markings = [{
        "id": c, "name": NAMES.get(c, c.replace("_", " ")), "score": round(s),
        "top10": [round(float(x), 3) for x in AVG[c].values],
        "band": [round(float(x), 3) for x in STD[c].values],
        "you": [round(float(x), 3) for x in new[c].values],
    } for c, s in sorted(scores.items(), key=lambda x: -x[1])]
    return {"overall": round(overall), "opener": opener, "tips": tips,
            "strength": strength, "markings": markings, "is_demo": False}
