from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import joblib, os, math, traceback, json
import numpy as np
import subprocess, sys, importlib

# Try to import requests or install at runtime (Render-friendly)
try:
    import requests
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])  # best-effort
        import requests
    except Exception:
        requests = None

# Try to import xgboost but don't fail if unavailable
try:
    import xgboost as xgb
except Exception:
    xgb = None

APP_TITLE = "Urban Vigil - Bengaluru Crime Predictor"
MODEL_PATH = "crime_predictor_bangalore_new.pkl"  # keep your model file here (optional)

# Safe ensure pkg_resources (setuptools) exists on older/newer pythons
try:
    import pkg_resources
except Exception:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools>=70.0.0"])
        import pkg_resources
    except Exception:
        pkg_resources = None

app = FastAPI(title=APP_TITLE)

# Allow all origins for university project (NOT for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Small set of Bengaluru places and coordinates
PLACE_COORDS = {
    "banashankari": (12.9250, 77.5670),
    "koramangala": (12.935222, 77.624481),
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
    """Attempt to load a model with joblib. If it fails, we keep model=None and rely on heuristics.
    Avoid calling sklearn internals that might trigger compatibility issues.
    """
    global model, model_info
    if not os.path.exists(MODEL_PATH):
        print(f"Model file not found at {MODEL_PATH} — continuing without model.")
        model = None
        model_info = {"loaded": False, "name": None, "n_features_in": None, "classes": None}
        return

    try:
        m = joblib.load(MODEL_PATH)
        # do not try to call super().get_params() or other fragile internals
        nfi = getattr(m, "n_features_in_", None) or getattr(m, "n_features_in", None)
        classes = getattr(m, "classes_", None)
        model = m
        model_info.update({"loaded": True, "name": type(m).__name__, "n_features_in": nfi, "classes": list(classes) if classes is not None else None})
        print("Model loaded safely:", model_info)
    except Exception as e:
        print("Failed to load model (safe mode). Error:", e)
        model = None
        model_info = {"loaded": False, "name": None, "n_features_in": None, "classes": None}


@app.on_event("startup")
def startup_event():
    safe_load_model()


# ---------------- Pydantic models ----------------
class PredictRequest(BaseModel):
    place: str = Field(..., description="Place / locality name (example: koramangala)")
    hour: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)
    extra: Optional[Dict[str, Any]] = None


class PredictResponse(BaseModel):
    place: str
    predicted_crime: str
    confidence: float
    risk_score: float
    details: Optional[Dict[str, Any]] = None


# ---------------- helpers ----------------

def heuristic_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
    """Fallback deterministic heuristic. Returns predicted, confidence(0-100), risk_score(0-100)"""
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


def try_model_predict(place: str, hour: Optional[int], day_of_week: Optional[int]):
    """If a model is loaded, try to predict. Any exception will be propagated to caller so they can fallback.
    The code carefully guards calls to possibly incompatible APIs.
    """
    if model is None:
        raise RuntimeError("No model loaded")

    # Build a simple feature vector: keep it defensive.
    n = model_info.get("n_features_in")
    if n and isinstance(n, int) and n > 0:
        vec = np.zeros(n, dtype=float)
        if n >= 1 and hour is not None:
            vec[0] = float(hour)
        if n >= 2 and day_of_week is not None:
            vec[1] = float(day_of_week)
        X = vec.reshape(1, -1)
    else:
        X = np.array([[hour if hour is not None else 0, day_of_week if day_of_week is not None else 0]])

    try:
        # Try predict_proba first (common). Wrap in try/except.
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            top_idx = int(np.argmax(probs, axis=1)[0])
            prob = float(probs[0, top_idx])
            pred = None
            if hasattr(model, "classes_"):
                try:
                    pred = model.classes_[top_idx]
                except Exception:
                    pred = str(top_idx)
        else:
            pred = model.predict(X)
            prob = 0.7
        if pred is None:
            pred = model.predict(X)
        predicted = str(pred[0]) if isinstance(pred, (list, np.ndarray)) else str(pred)
        confidence = float(prob * 100)
        risk_score = min(100.0, max(0.0, confidence))
        return predicted, confidence, risk_score
    except Exception as e:
        # bubble up
        raise RuntimeError(f"Model prediction failed: {e}")


# haversine distance
def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))
    return R * c


# ---------------- endpoints ----------------
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
        details = {"used_model": False, "error": str(e), "trace": traceback.format_exc().splitlines()[-8:]}
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
    top3 = h[:3]
    return {"places_monitored": len(h), "average_risk": round(avg_risk, 2), "top_3_risky": top3}


class SafeRouteRequest(BaseModel):
    src: str
    dst: str


@app.post("/safe-route")
def safe_route(req: SafeRouteRequest):
    src = req.src.strip().lower()
    dst = req.dst.strip().lower()
    if src not in PLACE_COORDS or dst not in PLACE_COORDS:
        raise HTTPException(status_code=400, detail="src/dst must be one of known places")
    if src == dst:
        return {"route": [ {"lat": PLACE_COORDS[src][0], "lng": PLACE_COORDS[src][1] } ], "message": "You're already there", "estimated_risk": 0.0}

    src_coord = PLACE_COORDS[src]
    dst_coord = PLACE_COORDS[dst]

    # If requests not available, fallback to simple route of waypoints
    use_osrm = requests is not None
    osrm_used = False
    osrm_route_coords = []
    sampled_points = []

    if use_osrm:
        try:
            # Build coordinates string in lon,lat order
            coord_str = f"{src_coord[1]},{src_coord[0]};{dst_coord[1]},{dst_coord[0]}"
            osrm_url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}?overview=full&geometries=geojson&steps=false"
            r = requests.get(osrm_url, timeout=8)
            jr = r.json()
            if jr.get("code") == "Ok" and jr.get("routes"):
                osrm_used = True
                geom = jr["routes"][0]["geometry"]
                coords = geom.get("coordinates", [])
                # convert to list of {lat,lng}
                osrm_route_coords = [{"lat": c[1], "lng": c[0]} for c in coords]
        except Exception as e:
            print("OSRM call failed, will fallback to straight route. Error:", e)

    # If OSRM succeeded, sample points along route every N steps to estimate risk
    def sample_and_score(coords_list: List[Dict[str, float]], step: int = 30):
        out = []
        for i, p in enumerate(coords_list):
            if i % step == 0 or i == len(coords_list) - 1:
                lat, lng = p["lat"], p["lng"]
                # find nearest place and use its heuristic/model prediction as proxy
                nearest_place = None
                nearest_d = 1e9
                for place, (plat, plon) in PLACE_COORDS.items():
                    d = haversine((lat, lng), (plat, plon))
                    if d < nearest_d:
                        nearest_d = d
                        nearest_place = place
                try:
                    _, _, r = try_model_predict(nearest_place, None, None)
                except Exception:
                    _, _, r = heuristic_predict(nearest_place, None, None)
                out.append({"lat": lat, "lng": lng, "risk": round(float(r), 2)})
        return out

    if osrm_used and osrm_route_coords:
        sampled_points = sample_and_score(osrm_route_coords, step= max(1, len(osrm_route_coords)//25))
        # estimate average risk along sampled points
        est_r = sum(p["risk"] for p in sampled_points) / max(1, len(sampled_points))
        return {"route": osrm_route_coords, "estimated_risk": round(est_r, 2), "note": "Route computed via OSRM and risk sampled along route", "osrm_used": True, "sampled_points": sampled_points}

    # Fallback: direct polyline between src/dst with optional waypoint insertion
    # compute simple waypoint candidates (existing places) to minimize risk
    base_dist = max(0.001, haversine(src_coord, dst_coord))

    def get_r(place):
        try:
            _, _, r = try_model_predict(place, None, None)
        except Exception:
            _, _, r = heuristic_predict(place, None, None)
        return float(r)

    risk_src = get_r(src)
    risk_dst = get_r(dst)
    direct_risk = (risk_src + risk_dst) / 2.0
    if direct_risk < 55:
        # return simple coordinate route
        return {"route": [{"lat": src_coord[0], "lng": src_coord[1]}, {"lat": dst_coord[0], "lng": dst_coord[1]}], "estimated_risk": round(direct_risk, 2), "note": "Direct route suggested (low risk)", "osrm_used": False}

    # Evaluate candidate waypoints
    candidates = []
    for place, coord in PLACE_COORDS.items():
        if place in (src, dst):
            continue
        d1 = haversine(src_coord, coord)
        d2 = haversine(coord, dst_coord)
        detour_ratio = (d1 + d2) / base_dist
        r = get_r(place)
        candidates.append({"place": place, "coord": coord, "risk": r, "detour_ratio": detour_ratio})

    alpha = 0.8
    beta = 0.2
    max_risk = max([c["risk"] for c in candidates] + [risk_src, risk_dst, 1])
    scored = []
    for c in candidates:
        norm_r = c["risk"] / max_risk
        norm_detour = min(c["detour_ratio"], 3.0) / 3.0
        score = alpha * norm_r + beta * norm_detour
        scored.append((score, c))
    scored_sorted = sorted(scored, key=lambda x: x[0])

    for score, c in scored_sorted:
        if c["detour_ratio"] <= 2.0:
            waypoint = c["place"]
            wcoord = c["coord"]
            est_risk = round((risk_src + c["risk"] + risk_dst) / 3.0, 2)
            route_coords = [ {"lat": src_coord[0], "lng": src_coord[1]}, {"lat": wcoord[0], "lng": wcoord[1]}, {"lat": dst_coord[0], "lng": dst_coord[1]} ]
            # sample each waypoint as sampled_points
            sampled_points = []
            for rc in route_coords:
                # nearest place proxy
                nearest_place = None
                nearest_d = 1e9
                for place, (plat, plon) in PLACE_COORDS.items():
                    d = haversine((rc["lat"], rc["lng"]), (plat, plon))
                    if d < nearest_d:
                        nearest_d = d
                        nearest_place = place
                try:
                    _, _, r = try_model_predict(nearest_place, None, None)
                except Exception:
                    _, _, r = heuristic_predict(nearest_place, None, None)
                sampled_points.append({"lat": rc["lat"], "lng": rc["lng"], "risk": round(float(r), 2)})
            return {"route": route_coords, "estimated_risk": est_risk, "waypoint": waypoint, "detour_ratio": round(c["detour_ratio"], 2), "note": "Waypoint inserted to reduce risk", "osrm_used": False, "sampled_points": sampled_points}

    # final fallback: direct coordinates
    return {"route": [{"lat": src_coord[0], "lng": src_coord[1]}, {"lat": dst_coord[0], "lng": dst_coord[1]}], "estimated_risk": round(direct_risk, 2), "note": "No suitable low-risk waypoint found; direct route provided", "osrm_used": False}


@app.get("/")
def root():
    return {"status": "ok", "message": APP_TITLE, "endpoints": ["/predict", "/heatmap", "/risky-zones", "/dashboard", "/safe-route", "/place-list", "/health"]}
