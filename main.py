
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import joblib, os, math, traceback
import numpy as np

# --- Fix for Render: ensure pkg_resources exists before loading model ---
import importlib, subprocess, sys

try:
    import pkg_resources
except ModuleNotFoundError:
    print("pkg_resources not found, installing setuptools...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"])
        pkg_resources = importlib.import_module("pkg_resources")
        print("pkg_resources imported successfully.")
    except Exception as e:
        print(f"Failed to import pkg_resources dynamically: {e}")
# -----------------------------------------------------------------------

APP_TITLE = "Urban Vigil - Bengaluru Crime Predictor"

app = FastAPI(title=APP_TITLE)

# Allow all origins (university project). In production narrow this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "crime_predictor_bangalore_new.pkl"

# Minimal place coordinates for Bengaluru (latitude, longitude)
PLACE_COORDS = {
    "banashankari": (12.9250, 77.5670),
    "koramangala": (12.9352, 77.6245),
    "indiranagar": (12.9719, 77.6412),
    "mg_road": (12.9763, 77.5995),
    "jayanagar": (12.9250, 77.5938),
    "hebbal": (13.0350, 77.5970),
    "whitefield": (12.9698, 77.7490),
    "electronic_city": (12.8419, 77.6605),
    "kammanahalli": (13.0101, 77.6197),
    "rajajinagar": (13.0136, 77.5551)
}

model = None
model_info = {"loaded": False, "name": None, "n_features_in": None, "classes": None}

def safe_load_model():
    global model, model_info
    if not os.path.exists(MODEL_PATH):
        return
        
    try:
        m = joblib.load(MODEL_PATH)
        # Try to introspect classifier attributes commonly available
        nfi = getattr(m, "n_features_in_", None)
        classes = getattr(m, "classes_", None)
        model = m
        model_info = {"loaded": True, "name": type(m).__name__, "n_features_in": nfi, "classes": list(classes) if classes is not None else None}
        print("Model loaded:", model_info)
    except Exception as e:
        print("Failed to load model, will use fallback heuristic. Error:", e)
        model = None

@app.on_event("startup")
def startup_event():
    safe_load_model()

# ---------------------------
# Pydantic models
# ---------------------------
class PredictRequest(BaseModel):
    place: str = Field(..., description="Place / locality name (example: banashankari)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day 0-23")
    day_of_week: Optional[int] = Field(None, ge=0, le=6, description="0=Monday ... 6=Sunday")
    extra: Optional[Dict[str, Any]] = None

class PredictResponse(BaseModel):
    place: str
    predicted_crime: str
    confidence: float = Field(..., ge=0.0, le=100.0)
    risk_score: float = Field(..., ge=0.0, le=100.0)
    details: Optional[Dict[str, Any]] = None

# ---------------------------
# Helpers
# ---------------------------
def heuristic_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
    """
    Simple fallback heuristic:
    - Certain hours (20-3) are higher risk.
    - Some localities have baseline risk multipliers.
    - Returns predicted crime (string), confidence (0-100), risk_score (0-100)
    """
    place_key = place.lower().replace(" ", "_")
    base = 30
    high_risk_places = {"mg_road", "whitefield", "electronic_city", "banashankari"}
    if place_key in high_risk_places:
        base += 20
    if hour is not None:
        if hour >= 20 or hour <= 3:
            base += 25
        elif 6 <= hour <= 9:
            base += 5
    if day_of_week is not None and day_of_week >=5:
        base += 5  # weekend slightly higher
    base = min(max(base, 5), 95)
    predicted = "Theft" if base < 50 else "Assault" if base < 75 else "Robbery"
    confidence = float(min(95, base + 5))
    risk_score = float(base)
    return predicted, confidence, risk_score

def try_model_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
    """
    Try to use the loaded model. We build a feature vector of length model.n_features_in if available.
    If the model cannot be used safely, raise Exception to let caller fall back.
    """
    if model is None:
        raise RuntimeError("No model loaded")
    n = model_info.get("n_features_in") or None
    X = None
    if n is not None:
        vec = np.zeros(n, dtype=float)
        if n >= 1 and hour is not None:
            vec[0] = float(hour)
        if n >= 2 and day_of_week is not None:
            vec[1] = float(day_of_week)
        X = vec.reshape(1, -1)
    else:
        X = np.array([[hour if hour is not None else 0, day_of_week if day_of_week is not None else 0]])
    if not hasattr(model, "predict_proba"):
        pred = model.predict(X)
        prob = None
    else:
        probs = model.predict_proba(X)
        top_idx = np.argmax(probs, axis=1)[0]
        prob = float(probs[0, top_idx])
        pred = model.classes_[top_idx] if hasattr(model, "classes_") else str(top_idx)
    predicted = str(pred[0]) if isinstance(pred, (list, np.ndarray)) else str(pred)
    confidence = float(prob*100) if prob is not None else 50.0
    risk_score = min(100.0, max(0.0, confidence))
    return predicted, confidence, risk_score

# ---------------------------
# Endpoints
# ---------------------------

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_info["loaded"], "model": model_info}

@app.get("/place-list")
def get_place_list():
    return {"places": list(PLACE_COORDS.keys()), "total": len(PLACE_COORDS)}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    place = req.place.strip().lower()
    if place not in PLACE_COORDS:
        raise HTTPException(status_code=400, detail=f"Unknown place '{req.place}'. Try one of: {', '.join(PLACE_COORDS.keys())}")
    try:
        predicted, confidence, risk_score = try_model_predict(place, req.hour, req.day_of_week)
        details = {"used_model": model_info["loaded"]}
    except Exception as e:
        predicted, confidence, risk_score = heuristic_predict(place, req.hour, req.day_of_week)
        details = {"used_model": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-5:]}
    return PredictResponse(place=place, predicted_crime=predicted, confidence=round(confidence,2), risk_score=round(risk_score,2), details=details)

@app.get("/heatmap")
def heatmap(hour: Optional[int]=None, day_of_week: Optional[int]=None):
    """
    Returns a simple heatmap JSON: list of {place, lat, lon, risk_score}
    """
    heat = []
    for place, (lat, lon) in PLACE_COORDS.items():
        try:
            predicted, confidence, risk_score = try_model_predict(place, hour, day_of_week)
        except Exception:
            _, _, risk_score = heuristic_predict(place, hour, day_of_week)
        heat.append({"place": place, "lat": lat, "lon": lon, "risk_score": round(float(risk_score),2)})
    heat_sorted = sorted(heat, key=lambda x: x["risk_score"], reverse=True)
    return {"heatmap": heat_sorted}

@app.get("/risky-zones")
def risky_zones(top: int = 5, hour: Optional[int]=None, day_of_week: Optional[int]=None):
    h = heatmap(hour=hour, day_of_week=day_of_week)["heatmap"]
    return {"top": top, "risky_zones": h[:top]}

@app.get("/dashboard")
def dashboard(hour: Optional[int]=None, day_of_week: Optional[int]=None):
    h = heatmap(hour=hour, day_of_week=day_of_week)["heatmap"]
    avg_risk = sum(item["risk_score"] for item in h) / max(1, len(h))
    top3 = h[:3]
    return {"places_monitored": len(h), "average_risk": round(avg_risk,2), "top_3_risky": top3}

class SafeRouteRequest(BaseModel):
    src: str
    dst: str

def haversine(a, b):
    # a, b are (lat, lon)
    R = 6371.0  # km
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))
    return R * c

@app.post("/safe-route")
def safe_route(req: SafeRouteRequest):
    """
    Improved safe-route:
    - Compute direct distance src->dst and risk.
    - Evaluate candidate waypoints: score = alpha*estimated_risk + beta*detour_ratio
      where detour_ratio = (src->waypoint + waypoint->dst) / (src->dst)
    - Choose waypoint minimizing score but with detour_ratio <= 2.0 (avoid massive detours)
    """
    src = req.src.strip().lower()
    dst = req.dst.strip().lower()
    if src not in PLACE_COORDS or dst not in PLACE_COORDS:
        raise HTTPException(status_code=400, detail="src/dst must be one of known places")
    if src == dst:
        return {"route": [src], "message": "You're already there", "estimated_risk": 0.0}

    src_coord = PLACE_COORDS[src]
    dst_coord = PLACE_COORDS[dst]
    base_dist = max(0.001, haversine(src_coord, dst_coord))  # km

    # get risks
    def get_r(place):
        try:
            _,_,r = try_model_predict(place, None, None)
        except Exception:
            _,_,r = heuristic_predict(place, None, None)
        return float(r)

    risk_src = get_r(src)
    risk_dst = get_r(dst)
    direct_risk = (risk_src + risk_dst) / 2.0

    # if direct route is already safe enough, return it
    if direct_risk < 55:
        return {"route": [src, dst], "estimated_risk": round(direct_risk,2), "note": "Direct route suggested (low risk)"}

    candidates = []
    for place, coord in PLACE_COORDS.items():
        if place in (src, dst):
            continue
        # distances
        d1 = haversine(src_coord, coord)
        d2 = haversine(coord, dst_coord)
        detour_ratio = (d1 + d2) / base_dist if base_dist>0 else float('inf')
        r = get_r(place)
        candidates.append({"place": place, "coord": coord, "risk": r, "d1": d1, "d2": d2, "detour_ratio": detour_ratio})

    # scoring: prefer low risk and small detour
    alpha = 0.8  # weight for risk
    beta = 0.2   # weight for detour
    # normalize risk and detour to comparable scales
    max_risk = max([c["risk"] for c in candidates] + [risk_src, risk_dst, 1])
    scored = []
    for c in candidates:
        norm_r = c["risk"] / max_risk
        norm_detour = min(c["detour_ratio"], 3.0) / 3.0  # cap
        score = alpha * norm_r + beta * norm_detour
        scored.append((score, c))

    scored_sorted = sorted(scored, key=lambda x: x[0])
    # pick best that doesn't detour too much
    for score, c in scored_sorted:
        if c["detour_ratio"] <= 2.0:
            waypoint = c["place"]
            est_risk = round((risk_src + c["risk"] + risk_dst)/3.0, 2)
            return {"route": [src, waypoint, dst], "estimated_risk": est_risk, "waypoint": waypoint, "detour_ratio": round(c["detour_ratio"],2), "note": "Waypoint inserted to reduce risk"}
    # fallback: return direct
    return {"route": [src, dst], "estimated_risk": round(direct_risk,2), "note": "No suitable low-risk waypoint found; direct route provided"}

# Root
@app.get("/")
def root():
    return {"status": "ok", "message": APP_TITLE, "endpoints": ["/predict", "/heatmap", "/risky-zones", "/dashboard", "/safe-route", "/place-list", "/health"]}
