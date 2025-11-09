
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib, os, math, json, requests
import numpy as np
import pandas as pd
from datetime import datetime

APP_TITLE = "Urban Vigil Pro - API (OSRM Integrated)"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_PATH = os.path.join(MODELS_DIR, os.getenv("MODEL_FILE", "Final_model.pkl"))
PLACE_LOOKUP = os.path.join(MODELS_DIR, "place_lookup.csv")
DATA_CSV = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

app = FastAPI(title=APP_TITLE)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

model, place_lookup_df, crime_df, location_cache = None, None, None, {}

# ------------------------ UTILITIES ------------------------
def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
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

def safe_loads():
    global model, place_lookup_df, crime_df, location_cache
    try:
        model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        print("Model loaded:", type(model).__name__ if model else "None")
    except Exception as e:
        print("Model load failed:", e)

    try:
        place_lookup_df = pd.read_csv(PLACE_LOOKUP) if os.path.exists(PLACE_LOOKUP) else None
        print("Loaded places:", len(place_lookup_df) if place_lookup_df is not None else 0)
    except Exception as e:
        print("Place lookup failed:", e)
        place_lookup_df = None

    try:
        if os.path.exists(DATA_CSV):
            df = pd.read_csv(DATA_CSV)
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            crime_df = df
            build_cache(df)
            print("Crime data loaded:", len(df))
    except Exception as e:
        print("Crime data failed:", e)

def build_cache(df):
    global location_cache
    df['lat_bin'] = (df['latitude'] * 100).round() / 100
    df['lon_bin'] = (df['longitude'] * 100).round() / 100
    grouped = df.groupby(['lat_bin', 'lon_bin']).agg({'type': ['count', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown']}).reset_index()
    grouped.columns = ['lat', 'lon', 'crime_count', 'most_common']
    location_cache = {f"{r.lat:.4f},{r.lon:.4f}": {"lat": float(r.lat), "lon": float(r.lon), "crime_count": int(r.crime_count), "most_common": str(r.most_common)} for _, r in grouped.iterrows()}

@app.on_event("startup")
def startup_event():
    safe_loads()

# ------------------------ API ENDPOINTS ------------------------
@app.post("/api/safe-route")
def safe_route(payload: Dict[str, Any]):
    src, dst = payload.get("src"), payload.get("dst")
    if not src or not dst:
        raise HTTPException(status_code=400, detail="src and dst are required")
    def resolve_place(p):
        if isinstance(p, dict) and 'lat' in p and 'lon' in p:
            return float(p['lat']), float(p['lon'])
        if isinstance(p, str) and place_lookup_df is not None:
            row = place_lookup_df[place_lookup_df['Place'].str.lower() == p.lower()]
            if not row.empty:
                return float(row.iloc[0]['Latitude']), float(row.iloc[0]['Longitude'])
        raise HTTPException(status_code=400, detail=f"Cannot resolve {p}")
    s, d = resolve_place(src), resolve_place(dst)

    osrm_url = f"https://router.project-osrm.org/route/v1/driving/{s[1]},{s[0]};{d[1]},{d[0]}?overview=full&geometries=geojson"
    try:
        resp = requests.get(osrm_url, timeout=10)
        data = resp.json()
        if "routes" in data and len(data["routes"]) > 0:
            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]
            geojson = route["geometry"]
            dist_km = route["distance"] / 1000.0
            duration_min = route["duration"] / 60.0
            # Compute risk along route
            samples = [{"lat": c[1], "lon": c[0], "risk": calculate_risk_score(c[1], c[0])} for c in coords[::max(1, len(coords)//50)]]
            avg_risk = round(sum(p["risk"] for p in samples) / len(samples), 1)
            return {"geometry": geojson, "average_risk": avg_risk, "distance_km": round(dist_km, 2), "duration_min": round(duration_min, 1), "samples": samples}
    except Exception as e:
        print("OSRM failed, fallback:", e)
    # fallback: straight line sample
    n = 30
    pts = [{"lat": s[0] + i / n * (d[0] - s[0]), "lon": s[1] + i / n * (d[1] - s[1]), "risk": calculate_risk_score(s[0] + i / n * (d[0] - s[0]), s[1] + i / n * (d[1] - s[1]))} for i in range(n + 1)]
    return {"route_points": pts, "average_risk": round(sum(p["risk"] for p in pts) / len(pts), 1), "distance_km": round(haversine(s, d), 2), "duration_min": None}
