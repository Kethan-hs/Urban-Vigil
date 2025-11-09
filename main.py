from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import joblib, os, math, json, requests
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================
# 🚀 URBAN VIGIL PRO — Full Intelligence Backend (Final Build)
# ============================================================

APP_TITLE = "Urban Vigil Pro - AI Safety API"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "Final_model_fixed.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
MAPPING_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
DATA_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ------------------------------------------------------------
# Global state
# ------------------------------------------------------------
model, encoders, crime_mapping, df, lookup_df, location_cache = None, {}, {}, None, None, {}

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))

def risk_label(score):
    if score >= 75:
        return "🔴 High Risk"
    elif score >= 50:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

def load_data():
    global model, encoders, crime_mapping, df, lookup_df, location_cache

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print("❌ Failed to load model:", e)

    # Load encoders
    try:
        encoders = joblib.load(ENCODER_PATH)
        print(f"✅ Encoders loaded: {list(encoders.keys())}")
    except Exception as e:
        print("⚠️ Failed to load encoders:", e)
        encoders = {}

    # Load mapping
    try:
        with open(MAPPING_PATH, "r") as f:
            crime_mapping = json.load(f)
        print(f"✅ Loaded {len(crime_mapping)} crime mappings.")
    except Exception as e:
        print("⚠️ Failed to load crime mapping:", e)
        crime_mapping = {}

    # Load lookup
    try:
        lookup_df = pd.read_csv(LOOKUP_PATH)
        print(f"✅ Loaded {len(lookup_df)} places from lookup.")
    except Exception as e:
        print("⚠️ Failed to load lookup file:", e)
        lookup_df = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])

    # Load main dataset
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df.dropna(subset=["latitude", "longitude"], inplace=True)
        build_cache(df)
        print(f"✅ Crime data loaded: {len(df)} records.")
    except Exception as e:
        print("⚠️ Failed to load crime dataset:", e)

def build_cache(df):
    global location_cache
    df["lat_bin"] = (df["latitude"] * 100).round() / 100
    df["lon_bin"] = (df["longitude"] * 100).round() / 100
    grouped = df.groupby(["lat_bin", "lon_bin"]).agg({
        "type": ["count", lambda x: x.mode()[0] if len(x) > 0 else "Unknown"]
    }).reset_index()
    grouped.columns = ["lat", "lon", "crime_count", "most_common"]
    location_cache = {
        f"{r.lat:.4f},{r.lon:.4f}": {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common)
        } for _, r in grouped.iterrows()
    }

@app.on_event("startup")
def startup_event():
    load_data()

# ------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "encoders_loaded": bool(encoders),
        "places": len(lookup_df),
        "records": len(df) if df is not None else 0,
        "clusters": len(location_cache)
    }

@app.get("/api/places")
def get_places():
    return lookup_df["Place"].dropna().drop_duplicates().sort_values().tolist()

@app.get("/api/predict")
def predict(place: str = Query(...), time: str = Query(...)):
    try:
        row = lookup_df[lookup_df["Place"].str.lower() == place.lower()]
        if row.empty:
            raise HTTPException(status_code=404, detail="Place not found")

        lat, lon = row.iloc[0]["Latitude"], row.iloc[0]["Longitude"]
        hour = datetime.strptime(time, "%H:%M").hour if ":" in time else int(time)
        day = datetime.now().strftime("%A")
        part_of_day = (
            "Morning" if 5 <= hour < 12 else
            "Afternoon" if 12 <= hour < 17 else
            "Evening" if 17 <= hour < 21 else
            "Night"
        )

        features = {
            "Latitude": lat,
            "Longitude": lon,
            "Hour": hour,
            "DayOfWeek": encoders["DayOfWeek"].transform([day])[0] if "DayOfWeek" in encoders else 0,
            "PartOfDay": encoders["PartOfDay"].transform([part_of_day])[0] if "PartOfDay" in encoders else 0,
            "Place": encoders["Place"].transform([place])[0] if "Place" in encoders else 0
        }

        X = pd.DataFrame([features])
        probs = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = crime_mapping.get(str(pred_idx), "Unknown")
        confidence = round(float(np.max(probs)) * 100, 2)
        risk = risk_label(confidence)

        return {
            "place": place,
            "time": time,
            "predicted_crime": pred_label,
            "confidence": f"{confidence}%",
            "risk_level": risk,
            "message": f"Prediction based on Bengaluru ML model ({model.__class__.__name__})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/heatmap")
def heatmap(limit: int = 150):
    heatmap_data = []
    for v in list(location_cache.values())[:limit]:
        risk = min(100, v["crime_count"] * 1.2)
        heatmap_data.append({
            "lat": v["lat"],
            "lon": v["lon"],
            "intensity": risk,
            "most_common": v["most_common"]
        })
    return {"heatmap": heatmap_data}

@app.post("/api/safe-route")
def safe_route(payload: dict):
    src, dst = payload.get("src"), payload.get("dst")
    if not src or not dst:
        raise HTTPException(status_code=400, detail="Source and destination required")
    try:
        osrm_url = f"https://router.project-osrm.org/route/v1/driving/{src['lon']},{src['lat']};{dst['lon']},{dst['lat']}?overview=full&geometries=geojson"
        resp = requests.get(osrm_url, timeout=10)
        data = resp.json()
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]
            samples = [{"lat": c[1], "lon": c[0], "risk": min(100, c[1] * 0.1)} for c in coords[::max(1, len(coords)//40)]]
            avg_risk = round(sum(s["risk"] for s in samples) / len(samples), 1)
            return {
                "geometry": route["geometry"],
                "distance_km": round(route["distance"]/1000, 2),
                "duration_min": round(route["duration"]/60, 1),
                "average_risk": avg_risk,
                "samples": samples
            }
        raise HTTPException(status_code=500, detail="OSRM route not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Route generation failed: {e}")

@app.get("/api/dashboard")
def dashboard():
    if df is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded")
    recent = df.tail(500)
    crime_counts = recent["type"].value_counts().to_dict()
    top_places = df["place"].value_counts().head(10).to_dict()
    return {
        "total_records": len(df),
        "recent_sample": len(recent),
        "top_crimes": crime_counts,
        "top_places": top_places
    }
