# ============================================================
# 🚀 URBAN VIGIL PRO — Final Production Backend (Render Ready)
# ============================================================

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib, os, math, json, requests
import numpy as np
import pandas as pd
from datetime import datetime

# ------------------------------------------------------------
# 🧠 App Configuration
# ------------------------------------------------------------
APP_TITLE = "Urban Vigil Pro - API (OSRM + Place-Aware)"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, os.getenv("MODEL_FILE", "Final_model_fixed.pkl"))
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_DATA_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ------------------------------------------------------------
# 🔧 Globals
# ------------------------------------------------------------
model, encoders, place_lookup_df, crime_df, location_cache, crime_mapping = None, None, None, None, {}, {}

# ------------------------------------------------------------
# 🧮 Utilities
# ------------------------------------------------------------
def haversine(a, b):
    """Calculate distance between two lat/lon points in km."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
    """Compute a heuristic safety risk score."""
    base = 30.0
    nearest, min_dist = None, 1e9
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_dist:
            min_dist, nearest = d, v
    if nearest:
        base += min(40, nearest['crime_count'] * 0.5)
        t = str(nearest.get('most_common', '')).lower()
        if 'murder' in t or 'rape' in t:
            base += 25
        elif 'robbery' in t or 'assault' in t:
            base += 12
        elif 'theft' in t or 'burglary' in t:
            base += 8
    if hour is not None:
        if hour >= 22 or hour <= 4:
            base += 20
        elif 18 <= hour <= 21:
            base += 10
        elif 6 <= hour <= 9:
            base += 5
    if day_of_week is not None and day_of_week >= 5:
        base += 5
    if min_dist > 1.0:
        base += min(10, min_dist * 2)
    return max(0, min(100, round(base, 1)))

def build_cache(df):
    """Create location crime cache for nearby risk estimation."""
    global location_cache
    df['lat_bin'] = (df['Latitude'] * 100).round() / 100
    df['lon_bin'] = (df['Longitude'] * 100).round() / 100
    grouped = (
        df.groupby(['lat_bin', 'lon_bin'])
        .agg({'Type': ['count', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown']})
        .reset_index()
    )
    grouped.columns = ['lat', 'lon', 'crime_count', 'most_common']
    location_cache = {
        f"{r.lat:.4f},{r.lon:.4f}": {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common)
        }
        for _, r in grouped.iterrows()
    }

# ------------------------------------------------------------
# ⚙️ Data Loading
# ------------------------------------------------------------
def safe_loads():
    global model, encoders, place_lookup_df, crime_df, crime_mapping
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️ Model not loaded: {e}")
        model = None
    try:
        encoders = joblib.load(ENCODER_PATH)
        print("✅ Encoders loaded:", list(encoders.keys()))
    except Exception as e:
        print(f"⚠️ Failed to load encoders: {e}")
        encoders = {}
    try:
        with open(CRIME_MAP_PATH, "r", encoding="utf-8") as f:
            crime_mapping = json.load(f)
        print(f"✅ Loaded {len(crime_mapping)} crime mappings.")
    except Exception as e:
        print(f"⚠️ Failed to load crime mapping: {e}")
    try:
        place_lookup_df = pd.read_csv(PLACE_LOOKUP_PATH)
        print(f"✅ Loaded {len(place_lookup_df)} places from lookup.")
    except Exception as e:
        print(f"⚠️ Failed to load place lookup: {e}")
        place_lookup_df = pd.DataFrame()
    try:
        crime_df = pd.read_csv(CRIME_DATA_PATH)
        build_cache(crime_df)
        print(f"✅ Crime data loaded: {len(crime_df)} records.")
    except Exception as e:
        print(f"⚠️ Failed to load crime data: {e}")

@app.on_event("startup")
def startup_event():
    safe_loads()

# ------------------------------------------------------------
# 🌐 API ROUTES
# ------------------------------------------------------------
@app.get("/api")
def home():
    return {"status": "ok", "message": "Urban Vigil Pro API is live 🚀"}

@app.get("/api/place-list")
def place_list():
    return place_lookup_df["Place"].dropna().drop_duplicates().sort_values().tolist()

@app.get("/api/place-lookup")
def place_lookup():
    """Alias for frontend expecting /api/place-lookup"""
    return place_list()

@app.get("/api/predict")
def predict_api(
    place: Optional[str] = Query(None),
    time: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None)
):
    """Predict risk or crime type for a given place or GPS coordinate."""
    if not (place or (lat and lon)):
        raise HTTPException(status_code=400, detail="Provide either place+time or lat+lon")

    # Use current time if not provided
    now = datetime.now().strftime("%H:%M")
    query_time = time or now
    query_place = place

    # Handle lat/lon → place mapping
    if not query_place and lat and lon:
        nearest_idx = ((place_lookup_df["Latitude"] - lat)**2 + (place_lookup_df["Longitude"] - lon)**2).idxmin()
        query_place = place_lookup_df.iloc[nearest_idx]["Place"]

    # Prepare input for model
    hour = datetime.strptime(query_time, "%H:%M").hour
    part_of_day = (
        "Morning" if 5 <= hour < 12 else
        "Afternoon" if 12 <= hour < 17 else
        "Evening" if 17 <= hour < 21 else "Night"
    )
    day_name = datetime.now().strftime("%A")
    month = datetime.now().month

    try:
        input_data = {
            "Latitude": float(lat or place_lookup_df.loc[place_lookup_df["Place"] == query_place, "Latitude"].iloc[0]),
            "Longitude": float(lon or place_lookup_df.loc[place_lookup_df["Place"] == query_place, "Longitude"].iloc[0]),
            "Hour": hour,
            "DayOfWeek": encoders["DayOfWeek"].transform([day_name])[0] if "DayOfWeek" in encoders else 0,
            "Month": month,
            "PartOfDay": encoders["PartOfDay"].transform([part_of_day])[0] if "PartOfDay" in encoders else 0,
            "Police Station": 0,
            "Place": encoders["Place"].transform([query_place])[0] if "Place" in encoders else 0,
        }
        X = pd.DataFrame([input_data])
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = crime_mapping.get(str(pred_idx), "Unknown")
        confidence = round(float(np.max(probs)) * 100, 2)
        risk_score = calculate_risk_score(input_data["Latitude"], input_data["Longitude"], hour)
        return {
            "place": query_place,
            "time": query_time,
            "predicted_crime": pred_label,
            "confidence": confidence,
            "risk_score": risk_score,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

@app.post("/api/safe-route")
def safe_route(payload: Dict[str, Any]):
    src, dst = payload.get("src"), payload.get("dst")
    if not src or not dst:
        raise HTTPException(status_code=400, detail="src and dst required")
    def resolve(p):
        if isinstance(p, dict) and 'lat' in p and 'lon' in p:
            return float(p['lat']), float(p['lon'])
        if isinstance(p, str):
            row = place_lookup_df[place_lookup_df['Place'].str.lower() == p.lower()]
            if not row.empty:
                return float(row.iloc[0]['Latitude']), float(row.iloc[0]['Longitude'])
        raise HTTPException(status_code=400, detail=f"Cannot resolve {p}")
    s, d = resolve(src), resolve(dst)
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{s[1]},{s[0]};{d[1]},{d[0]}?overview=full&geometries=geojson"
        resp = requests.get(url, timeout=10).json()
        if "routes" in resp and resp["routes"]:
            route = resp["routes"][0]
            coords = route["geometry"]["coordinates"]
            samples = [{"lat": c[1], "lon": c[0], "risk": calculate_risk_score(c[1], c[0])} for c in coords[::max(1, len(coords)//50)]]
            avg_risk = round(sum(p["risk"] for p in samples) / len(samples), 1)
            return {
                "geometry": route["geometry"],
                "average_risk": avg_risk,
                "distance_km": round(route["distance"] / 1000, 2),
                "duration_min": round(route["duration"] / 60, 1),
                "samples": samples
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OSRM failed: {e}")

# ------------------------------------------------------------
# ✅ Run locally
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
