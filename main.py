from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import joblib, os, math, traceback, requests
import numpy as np
import importlib, subprocess, sys

APP_TITLE = "Urban Vigil - Bengaluru Crime Predictor"
MODEL_PATH = "crime_predictor_bangalore_new.pkl"

# ----------------------------------------
# Safe Import Handling for Render + Python 3.13
# ----------------------------------------
try:
    import pkg_resources
except ModuleNotFoundError:
    print("⚠️ pkg_resources missing, installing setuptools...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools>=70.0.0"])
        pkg_resources = importlib.import_module("pkg_resources")
        print("✅ pkg_resources imported successfully.")
    except Exception as e:
        print(f"❌ Failed to import pkg_resources dynamically: {e}")
# ----------------------------------------

app = FastAPI(title=APP_TITLE)

# Allow all origins (frontend, localhost, vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# Place Data (Bengaluru coordinates)
# ----------------------------------------
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

# ----------------------------------------
# MODEL LOADING (Robust fix for xgboost)
# ----------------------------------------
def safe_load_model():
    global model, model_info
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    try:
        m = joblib.load(MODEL_PATH)
        if hasattr(m, "get_booster"):
            try:
                _ = m.get_booster()
                print("✅ Booster extracted successfully.")
            except Exception as booster_err:
                print("⚠️ Booster extraction failed:", booster_err)

        nfi = getattr(m, "n_features_in_", None)
        classes = getattr(m, "classes_", None)

        model = m
        model_info.update({
            "loaded": True,
            "name": type(m).__name__,
            "n_features_in": nfi,
            "classes": list(classes) if classes is not None else None
        })
        print("✅ Model loaded:", model_info)

    except Exception as e:
        print(f"❌ Failed to load model, fallback enabled. Error: {e}")
        model = None
        model_info["loaded"] = False

@app.on_event("startup")
def startup_event():
    safe_load_model()

# ----------------------------------------
# Pydantic Models
# ----------------------------------------
class PredictRequest(BaseModel):
    place: str = Field(..., description="Place / locality name (example: banashankari)")
    hour: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    extra: Optional[Dict[str, Any]] = None

class PredictResponse(BaseModel):
    place: str
    predicted_crime: str
    confidence: float
    risk_score: float
    details: Optional[Dict[str, Any]] = None

class SafeRouteRequest(BaseModel):
    src: str
    dst: str

# ----------------------------------------
# Heuristic Predictor (Fallback)
# ----------------------------------------
def heuristic_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
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
    if day_of_week is not None and day_of_week >= 5:
        base += 5
    base = min(max(base, 5), 95)

    predicted = "Theft" if base < 50 else "Assault" if base < 75 else "Robbery"
    confidence = float(min(95, base + 5))
    risk_score = float(base)
    return predicted, confidence, risk_score

# ----------------------------------------
# Safe Model Predictor (fixed for xgboost)
# ----------------------------------------
def try_model_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
    if model is None:
        raise RuntimeError("Model not loaded.")

    n = model_info.get("n_features_in") or None
    if n:
        vec = np.zeros(n, dtype=float)
        if n >= 1 and hour is not None:
            vec[0] = float(hour)
        if n >= 2 and day_of_week is not None:
            vec[1] = float(day_of_week)
        X = vec.reshape(1, -1)
    else:
        X = np.array([[hour or 0, day_of_week or 0]])

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            top_idx = np.argmax(probs, axis=1)[0]
            prob = float(probs[0, top_idx])
            pred = model.classes_[top_idx] if hasattr(model, "classes_") else str(top_idx)
        else:
            pred = model.predict(X)
            prob = 0.7
        predicted = str(pred[0]) if isinstance(pred, (list, np.ndarray)) else str(pred)
        confidence = float(prob * 100)
        risk_score = min(100.0, max(0.0, confidence))
        return predicted, confidence, risk_score
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

# ----------------------------------------
# Helper: Haversine
# ----------------------------------------
def haversine(a, b):
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

# ----------------------------------------
# OSRM Integration for Safe Route
# ----------------------------------------
OSRM_BASE = "https://router.project-osrm.org"

def get_osrm_route(start_coord, end_coord):
    s_lon, s_lat = start_coord[1], start_coord[0]
    e_lon, e_lat = end_coord[1], end_coord[0]
    url = f"{OSRM_BASE}/route/v1/driving/{s_lon},{s_lat};{e_lon},{e_lat}"
    params = {"overview": "full", "geometries": "geojson", "steps": "false"}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "routes" not in data or not data["routes"]:
        raise RuntimeError("No routes returned by OSRM")
    coords = data["routes"][0]["geometry"]["coordinates"]
    return [(pt[1], pt[0]) for pt in coords]

def nearest_place_risk(lat, lng):
    best_place, best_dist, best_risk = None, float("inf"), None
    for place, (plat, plon) in PLACE_COORDS.items():
        d = haversine((lat, lng), (plat, plon))
        if d < best_dist:
            best_dist, best_place = d, place
    try:
        _, _, r = try_model_predict(best_place, None, None)
    except Exception:
        _, _, r = heuristic_predict(best_place, None, None)
    return float(r)

def compute_route_risk(route_coords, sample_every=10):
    if not route_coords:
        return 0.0, []
    sampled = route_coords[::sample_every] if len(route_coords) > sample_every else route_coords
    risks, sampled_with_risks = [], []
    for (lat, lon) in sampled:
        r = nearest_place_risk(lat, lon)
        risks.append(r)
        sampled_with_risks.append({"lat": lat, "lng": lon, "risk": r})
    avg_risk = float(sum(risks) / max(1, len(risks)))
    return avg_risk, sampled_with_risks

@app.post("/safe-route")
def safe_route(req: SafeRouteRequest):
    src, dst = req.src.strip().lower(), req.dst.strip().lower()
    if src not in PLACE_COORDS or dst not in PLACE_COORDS:
        raise HTTPException(status_code=400, detail="src/dst must be one of known places")
    if src == dst:
        return {"route": [{"lat": PLACE_COORDS[src][0], "lng": PLACE_COORDS[src][1]}],
                "message": "You're already there", "estimated_risk": 0.0, "osrm_used": False}

    src_coord, dst_coord = PLACE_COORDS[src], PLACE_COORDS[dst]

    try:
        route_coords = get_osrm_route(src_coord, dst_coord)
        avg_risk, sampled = compute_route_risk(route_coords, sample_every=max(1, len(route_coords)//50))
        route_payload = [{"lat": lat, "lng": lng} for (lat, lng) in route_coords]
        return {
            "route": route_payload,
            "estimated_risk": round(avg_risk, 2),
            "note": "Route computed via OSRM and risk sampled along route",
            "osrm_used": True,
            "sampled_points": sampled
        }
    except Exception as e:
        print(f"⚠️ OSRM failed, fallback: {e}")
        # fallback: direct haversine risk
        risk_src, risk_dst = nearest_place_risk(*src_coord), nearest_place_risk(*dst_coord)
        avg_risk = round((risk_src + risk_dst) / 2, 2)
        return {
            "route": [{"lat": src_coord[0], "lng": src_coord[1]}, {"lat": dst_coord[0], "lng": dst_coord[1]}],
            "estimated_risk": avg_risk,
            "note": "OSRM unavailable, fallback direct route",
            "osrm_used": False
        }

# ----------------------------------------
# API ENDPOINTS
# ----------------------------------------
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
    return PredictResponse(place=place, predicted_crime=predicted, confidence=round(confidence, 2), risk_score=round(risk_score, 2), details=details)

@app.get("/heatmap")
def heatmap(hour: Optional[int] = None, day_of_week: Optional[int] = None):
    heat = []
    for place, (lat, lon) in PLACE_COORDS.items():
        try:
            _, _, risk_score = try_model_predict(place, hour, day_of_week)
        except Exception:
            _, _, risk_score = heuristic_predict(place, hour, day_of_week)
        heat.append({"place": place, "lat": lat, "lon": lon, "risk_score": round(float(risk_score), 2)})
    return {"heatmap": sorted(heat, key=lambda x: x["risk_score"], reverse=True)}

@app.get("/risky-zones")
def risky_zones(top: int = 5, hour: Optional[int] = None, day_of_week: Optional[int] = None):
    h = heatmap(hour=hour, day_of_week=day_of_week)["heatmap"]
    return {"top": top, "risky_zones": h[:top]}

@app.get("/dashboard")
def dashboard(hour: Optional[int] = None, day_of_week: Optional[int] = None):
    h = heatmap(hour=hour, day_of_week=day_of_week)["heatmap"]
    avg_risk = sum(item["risk_score"] for item in h) / max(1, len(h))
    return {"places_monitored": len(h), "average_risk": round(avg_risk, 2), "top_3_risky": h[:3]}

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": APP_TITLE,
        "endpoints": ["/predict", "/heatmap", "/risky-zones", "/dashboard", "/safe-route", "/place-list", "/health"]
    }
