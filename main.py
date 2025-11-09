# remastered main.py — Urban Vigil Pro (production-ready)
# - robust loading, TTL caches, safe fallbacks, OSRM integration
# - keeps /api prefix, static serving if frontend built into repo

import os
import math
import json
import time
import logging
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# -------------------------
# Config
# -------------------------
APP_TITLE = "Urban Vigil Pro - API"
API_PREFIX = "/api"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATHS = [
    os.path.join(MODELS_DIR, os.getenv("MODEL_FILE", "Final_model_fixed.pkl")),
    os.path.join(MODELS_DIR, "Final_model.pkl"),
    os.path.join(MODELS_DIR, "Final_model_fixed.pkl"),
]
ENCODER_PATHS = [
    os.path.join(MODELS_DIR, os.getenv("ENCODERS_FILE", "label_encoders_fixed.pkl")),
    os.path.join(MODELS_DIR, "label_encoders.pkl"),
]
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_DATA_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")
OSRM_URL = os.getenv("OSRM_URL", "https://router.project-osrm.org")
CACHE_TTL = int(os.getenv("CACHE_TTL", "120"))  # seconds
HEATMAP_LIMIT = int(os.getenv("HEATMAP_LIMIT", "100"))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("urban-vigil")

# -------------------------
# App init
# -------------------------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock this down in prod to your frontend domain(s)
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Mount static if available (common build outputs)
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")
elif os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")

# -------------------------
# Globals & thread-safety
# -------------------------
_model = None
_encoders = {}
_crime_mapping: Dict[str, Any] = {}
_place_lookup_df: Optional[pd.DataFrame] = None
_crime_df: Optional[pd.DataFrame] = None
_location_cache: Dict[str, Dict[str, Any]] = {}
_load_lock = threading.Lock()

# Simple TTL cache decorator (thread-safe)
_cache_store: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def ttl_cache(key: str, ttl: int = CACHE_TTL):
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = key + ":" + json.dumps({"args": args, "kwargs": kwargs}, default=str)
            now = time.time()
            with _cache_lock:
                item = _cache_store.get(cache_key)
                if item and now - item["ts"] < ttl:
                    return item["value"]
            value = func(*args, **kwargs)
            with _cache_lock:
                _cache_store[cache_key] = {"value": value, "ts": time.time()}
            return value
        return wrapper
    return decorator


# -------------------------
# Utilities
# -------------------------
def haversine(a, b):
    """Distance km between (lat, lon) tuples."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))


def safe_transform(enc, col_name, value):
    """Safe transform wrapper for label encoders (returns 0 on failure)."""
    try:
        if enc and col_name in enc:
            transformer = enc[col_name]
            # transformer may be sklearn LabelEncoder or similar
            if hasattr(transformer, "transform"):
                return int(transformer.transform([value])[0])
    except Exception as e:
        logger.debug("Encoder transform failed for %s=%s (%s)", col_name, value, e)
    return 0


def detect_and_normalize_place_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure place lookup uses standardized columns: Place, Latitude, Longitude."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Place", "Latitude", "Longitude"])
    cols = [c.lower().strip() for c in df.columns]
    rename_map = {}
    # find place-like column
    for original, lower in zip(df.columns, cols):
        if any(k in lower for k in ["police", "station", "stat", "place"]):
            rename_map[original] = "Place"
            break
    # find lat/lon
    for original, lower in zip(df.columns, cols):
        if "lat" in lower:
            rename_map[original] = "Latitude"
        if "lon" in lower or "long" in lower:
            rename_map[original] = "Longitude"
    df = df.rename(columns=rename_map)
    # ensure required cols exist
    for c in ["Place", "Latitude", "Longitude"]:
        if c not in df.columns:
            df[c] = None
    # drop rows without coords
    df = df.dropna(subset=["Latitude", "Longitude"])
    # unify types
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"])
    df["Place"] = df["Place"].astype(str).str.strip()
    return df[["Place", "Latitude", "Longitude"]]


def build_location_cache_from_crime_df(df: pd.DataFrame):
    """Aggregate crime_df into location clusters for quick nearest lookup."""
    global _location_cache
    if df is None or df.empty:
        _location_cache = {}
        return
    df = df.copy()
    # normalize column names if present with differing case
    for c in df.columns:
        if c.lower() == "latitude":
            df = df.rename(columns={c: "Latitude"})
        if c.lower() == "longitude":
            df = df.rename(columns={c: "Longitude"})
        if c.lower() == "type":
            df = df.rename(columns={c: "Type"})
    if "Latitude" not in df.columns or "Longitude" not in df.columns:
        _location_cache = {}
        return
    df["lat_bin"] = (df["Latitude"] * 100).round() / 100
    df["lon_bin"] = (df["Longitude"] * 100).round() / 100
    grouped = (
        df.groupby(["lat_bin", "lon_bin"])
        .agg({"Type": ["count", lambda x: x.mode()[0] if len(x) else "Unknown"]})
        .reset_index()
    )
    # flatten columns
    grouped.columns = ["lat", "lon", "crime_count", "most_common"]
    cache = {}
    for _, r in grouped.iterrows():
        key = f"{r.lat:.4f},{r.lon:.4f}"
        cache[key] = {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common),
        }
    _location_cache = cache
    logger.info("Built location cache (%d clusters)", len(_location_cache))


def find_nearest_place_from_coords(lat: float, lon: float):
    """Return nearest place name from place_lookup_df using simple Euclidean (fast) or haversine."""
    global _place_lookup_df
    if _place_lookup_df is None or _place_lookup_df.empty:
        return None
    # squared distance over lat/lon (cheap) then pick nearest via haversine check
    d2 = ((_place_lookup_df["Latitude"] - lat) ** 2 + (_place_lookup_df["Longitude"] - lon) ** 2)
    idx = d2.idxmin()
    try:
        return _place_lookup_df.loc[idx, "Place"]
    except Exception:
        return None


# -------------------------
# Loading
# -------------------------
def safe_load_all():
    """Load model, encoders, mapping, CSVs. Use lock to avoid race on deployment."""
    global _model, _encoders, _crime_mapping, _place_lookup_df, _crime_df
    with _load_lock:
        # model
        for p in MODEL_PATHS:
            if os.path.exists(p):
                try:
                    _model = joblib.load(p)
                    logger.info("Model loaded from %s", p)
                    break
                except Exception as e:
                    logger.warning("Failed to load model from %s: %s", p, e)
        if _model is None:
            logger.warning("No model file found - model predictions will be disabled (historical fallback enabled)")

        # encoders
        for p in ENCODER_PATHS:
            if os.path.exists(p):
                try:
                    _encoders = joblib.load(p)
                    logger.info("Encoders loaded from %s -> keys: %s", p, list(_encoders.keys()))
                    break
                except Exception as e:
                    logger.warning("Failed to load encoders from %s: %s", p, e)
        if not _encoders:
            _encoders = {}

        # crime mapping
        try:
            if os.path.exists(CRIME_MAP_PATH):
                with open(CRIME_MAP_PATH, "r", encoding="utf-8") as fh:
                    _crime_mapping = json.load(fh)
                    logger.info("Crime mapping loaded (%d entries)", len(_crime_mapping))
        except Exception as e:
            logger.warning("Failed to load crime mapping: %s", e)
            _crime_mapping = {}

        # place lookup
        try:
            if os.path.exists(PLACE_LOOKUP_PATH):
                df = pd.read_csv(PLACE_LOOKUP_PATH)
                _place_lookup_df = detect_and_normalize_place_lookup(df)
                logger.info("Place lookup loaded (%d rows)", len(_place_lookup_df))
            else:
                _place_lookup_df = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])
                logger.warning("place_lookup.csv not found at %s", PLACE_LOOKUP_PATH)
        except Exception as e:
            logger.warning("Failed to load place lookup: %s", e)
            _place_lookup_df = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])

        # crime dataset
        try:
            if os.path.exists(CRIME_DATA_PATH):
                df = pd.read_csv(CRIME_DATA_PATH)
                # Standardize columns if they exist with different case / names
                _crime_df = df
                build_location_cache_from_crime_df(_crime_df)
                logger.info("Crime dataset loaded (%d rows)", len(_crime_df))
            else:
                _crime_df = pd.DataFrame()
                logger.warning("Crime data not found at %s", CRIME_DATA_PATH)
        except Exception as e:
            logger.exception("Failed to load crime data: %s", e)
            _crime_df = pd.DataFrame()


# Load once on startup
@app.on_event("startup")
def _startup():
    safe_load_all()


# -------------------------
# Schemas
# -------------------------
class PredictResponse(BaseModel):
    place: Optional[str]
    time: str
    predicted_crime: str
    confidence: float
    risk_score: float
    used_model: bool = False


class SafeRouteRequest(BaseModel):
    src: Any = Field(..., description="Either {'lat':..,'lon':..} or place name string")
    dst: Any = Field(..., description="Either {'lat':..,'lon':..} or place name string")


# -------------------------
# API: Health & Metadata
# -------------------------
@app.get(f"{API_PREFIX}/health")
def health():
    return {
        "status": "ok",
        "model_loaded": bool(_model),
        "place_lookup_count": len(_place_lookup_df) if _place_lookup_df is not None else 0,
        "crime_records": len(_crime_df) if _crime_df is not None else 0,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# -------------------------
# API: Place list / lookup
# -------------------------
@app.get(f"{API_PREFIX}/place-list")
@ttl_cache(key="place_list", ttl=300)
def place_list():
    global _place_lookup_df
    if _place_lookup_df is None:
        return []
    return _place_lookup_df["Place"].dropna().drop_duplicates().sort_values().tolist()


@app.get(f"{API_PREFIX}/place-lookup")
def place_lookup():
    """Return the whole lookup table — frontend expects a list; we return rows for convenience."""
    global _place_lookup_df
    if _place_lookup_df is None:
        return []
    return _place_lookup_df.to_dict(orient="records")


# -------------------------
# API: Heatmap (aggregated clusters)
# -------------------------
@app.get(f"{API_PREFIX}/heatmap")
@ttl_cache(key="heatmap", ttl=60)
def heatmap(limit: int = HEATMAP_LIMIT):
    """Return aggregated heatmap clusters. Each includes lat, lon, crime_count, risk_score."""
    if not _location_cache:
        return {"heatmap": [], "total": 0}
    heatmap = []
    for v in list(_location_cache.values())[:limit]:
        risk = calculate_risk_score(v["lat"], v["lon"])
        heatmap.append({
            "lat": v["lat"],
            "lon": v["lon"],
            "crime_count": v["crime_count"],
            "most_common": v.get("most_common", "Unknown"),
            "risk_score": risk
        })
    heatmap = sorted(heatmap, key=lambda r: r["risk_score"], reverse=True)
    return {"heatmap": heatmap, "total": len(heatmap)}


# -------------------------
# API: Predict (GET)
# -------------------------
@app.get(f"{API_PREFIX}/predict", response_model=PredictResponse)
def predict(
    place: Optional[str] = Query(None),
    time: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    """
    Predict crime (model) + risk score for a place or coordinates.
    - If model isn't available we return predicted_crime='Model unavailable' and still return risk_score from historical data.
    - Default time = current server time.
    """
    if not (place or (lat is not None and lon is not None)):
        raise HTTPException(status_code=400, detail="Provide either place or lat+lon")

    # Resolve place from lat/lon if place not provided
    used_model = False
    resolved_place = place
    if not resolved_place:
        resolved_place = find_nearest_place_from_coords(lat, lon) or None

    # determine time
    now = datetime.now()
    time_str = time or now.strftime("%H:%M")
    try:
        hour = datetime.strptime(time_str, "%H:%M").hour
    except Exception:
        hour = now.hour
        time_str = now.strftime("%H:%M")

    # Derive coordinates from place if needed
    coord_lat, coord_lon = None, None
    try:
        if lat is not None and lon is not None:
            coord_lat, coord_lon = float(lat), float(lon)
        elif resolved_place:
            row = _place_lookup_df[_place_lookup_df["Place"].str.lower() == str(resolved_place).lower()]
            if not row.empty:
                coord_lat, coord_lon = float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
    except Exception as e:
        logger.debug("Coordinate resolution error: %s", e)

    # Calculate risk score from historical clusters
    risk_score = calculate_risk_score(coord_lat or 0.0, coord_lon or 0.0, hour)

    # Attempt model prediction if available
    predicted_crime = "Model unavailable"
    confidence = 0.0
    if _model is not None:
        try:
            # Prepare model features - attempt to match naming in your model
            part_of_day = (
                "Morning" if 5 <= hour < 12 else
                "Afternoon" if 12 <= hour < 17 else
                "Evening" if 17 <= hour < 21 else "Night"
            )
            day_name = now.strftime("%A")
            month = now.month

            input_data = {
                "Latitude": float(coord_lat or 0.0),
                "Longitude": float(coord_lon or 0.0),
                "Hour": int(hour),
                "DayOfWeek": safe_transform(_encoders, "DayOfWeek", day_name) if _encoders else 0,
                "Month": int(month),
                "PartOfDay": safe_transform(_encoders, "PartOfDay", part_of_day) if _encoders else 0,
                "Police Station": 0,
                "Place": safe_transform(_encoders, "Place", resolved_place or "Unknown") if _encoders else 0
            }
            X = pd.DataFrame([input_data])
            if hasattr(_model, "predict_proba"):
                probs = _model.predict_proba(X)[0]
                idx = int(np.argmax(probs))
                predicted_crime = _crime_mapping.get(str(idx), str(idx)) if _crime_mapping else str(idx)
                confidence = float(np.max(probs) * 100)
            else:
                pred = _model.predict(X)[0]
                predicted_crime = str(pred)
                confidence = 75.0
            used_model = True
        except Exception as e:
            logger.exception("Model prediction failed: %s", e)
            predicted_crime = "Model failed"
            confidence = 0.0

    return {
        "place": resolved_place,
        "time": time_str,
        "predicted_crime": predicted_crime,
        "confidence": round(confidence, 2),
        "risk_score": round(risk_score, 1),
        "used_model": used_model,
    }


# -------------------------
# API: Safe route (POST)
# -------------------------
@app.post(f"{API_PREFIX}/safe-route")
def safe_route(req: SafeRouteRequest):
    """
    Accepts JSON: { "src": <place-string | {lat,lon}>, "dst": <place-string | {lat,lon}> }
    Returns geometry (geojson), average risk along route, distance_km, duration_min, samples (lat/lon/risk)
    """
    payload = req.dict()
    src = payload.get("src")
    dst = payload.get("dst")

    def resolve_point(p):
        # dict with lat/lon
        if isinstance(p, dict):
            if "lat" in p and "lon" in p:
                return float(p["lat"]), float(p["lon"])
            if "latitude" in p and "longitude" in p:
                return float(p["latitude"]), float(p["longitude"])
            raise HTTPException(status_code=400, detail=f"Invalid coordinate object: {p}")
        # string place
        if isinstance(p, str):
            row = _place_lookup_df[_place_lookup_df["Place"].str.lower() == p.lower()] if _place_lookup_df is not None else None
            if row is not None and not row.empty:
                return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
        raise HTTPException(status_code=400, detail=f"Cannot resolve point: {p}")

    s = resolve_point(src)
    d = resolve_point(dst)

    # call OSRM route API
    try:
        osrm_endpoint = f"{OSRM_URL}/route/v1/driving/{s[1]},{s[0]};{d[1]},{d[0]}?overview=full&geometries=geojson"
        resp = requests.get(osrm_endpoint, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]  # list of [lon, lat]
            # sample along route (max 50 points)
            step = max(1, int(len(coords) / 50))
            samples = []
            for c in coords[::step]:
                clat, clon = float(c[1]), float(c[0])
                samples.append({"lat": clat, "lon": clon, "risk": calculate_risk_score(clat, clon)})
            avg_risk = round(sum([samp["risk"] for samp in samples]) / len(samples), 1) if samples else 0.0
            return {
                "geometry": route["geometry"],
                "average_risk": avg_risk,
                "distance_km": round(route.get("distance", 0) / 1000.0, 2),
                "duration_min": round(route.get("duration", 0) / 60.0, 1),
                "samples": samples,
            }
    except requests.RequestException as e:
        logger.warning("OSRM request failed, falling back to straight-line: %s", e)

    # fallback: straight-line sample
    n = 30
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = s[0] + t * (d[0] - s[0])
        lon = s[1] + t * (d[1] - s[1])
        pts.append({"lat": lat, "lon": lon, "risk": calculate_risk_score(lat, lon)})
    avg_risk = round(sum(p["risk"] for p in pts) / len(pts), 1)
    return {"route_points": pts, "average_risk": avg_risk, "distance_km": round(haversine(s, d), 2), "duration_min": None}


# -------------------------
# API: Dashboard (summary)
# -------------------------
@app.get(f"{API_PREFIX}/dashboard")
@ttl_cache(key="dashboard", ttl=30)
def dashboard():
    """Return small summary used by frontend dashboard."""
    total_crimes = len(_crime_df) if _crime_df is not None else 0
    monitored_locations = len(_location_cache)
    heat = heatmap(limit=10)["heatmap"] if _location_cache else []
    top_risky = heat[:5]
    recent_30d = 0
    start_iso = end_iso = None
    if _crime_df is not None and "Date_fixed" in _crime_df.columns:
        try:
            _crime_df["datetime"] = pd.to_datetime(_crime_df["Date_fixed"], errors="coerce")
            recent_30d = int((_crime_df["datetime"] >= (datetime.now() - timedelta(days=30))).sum())
            start_iso = _crime_df["datetime"].min().isoformat() if not _crime_df["datetime"].isna().all() else None
            end_iso = _crime_df["datetime"].max().isoformat() if not _crime_df["datetime"].isna().all() else None
        except Exception:
            recent_30d = 0
    return {
        "total_crimes": total_crimes,
        "recent_crimes_30d": recent_30d,
        "top_risky_areas": top_risky,
        "monitored_locations": monitored_locations,
        "data_date_range": {"start": start_iso, "end": end_iso}
    }


# -------------------------
# Exception handlers (consistent JSON)
# -------------------------
@app.exception_handler(HTTPException)
def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(Exception)
def generic_exception_handler(request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# -------------------------
# Run locally
# -------------------------
if __name__ == "__main__":
    import uvicorn
    safe_load_all()
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
