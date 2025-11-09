# main.py
# Urban Vigil Pro - Production-ready backend (FastAPI)
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import joblib, os, math, json, requests, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---------- CONFIG ----------
APP_TITLE = "Urban Vigil Pro - API"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "Final_model_fixed.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_CSV_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

# ---------- APP ----------
app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Attempt to serve built frontend if present (common build output paths)
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
elif os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

# ---------- LOGGING ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("urban-vigil")

# ---------- GLOBALS ----------
model = None
encoders = {}
crime_mapping: Dict[str, Any] = {}
place_lookup_df: pd.DataFrame = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])
crime_df: Optional[pd.DataFrame] = None
location_cache: Dict[str, Dict[str, Any]] = {}
# location_cache keys: "lat,lon" -> {'lat':float,'lon':float,'crime_count':int,'most_common':str}

# ---------- UTILITIES ----------
def haversine(a, b):
    # a, b = (lat, lon)
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    x = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
    base = 30.0
    nearest, min_dist = None, float("inf")
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_dist:
            min_dist, nearest = d, v
    if nearest:
        base += min(40, nearest.get('crime_count', 0) * 0.5)
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
    if min_dist > 1.0 and min_dist != float("inf"):
        base += min(10, min_dist * 2)
    return max(0, min(100, round(base, 1)))

def build_cache(df: pd.DataFrame):
    global location_cache
    if df is None or df.empty:
        location_cache = {}
        return
    # normalize column names:
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ('type', 'crime', 'offence')), None)
    if lat_col is None or lon_col is None:
        logger.warning("No lat/lon columns found for cache build.")
        location_cache = {}
        return
    df = df.copy()
    df['lat_bin'] = (pd.to_numeric(df[lat_col], errors='coerce') * 100).round() / 100
    df['lon_bin'] = (pd.to_numeric(df[lon_col], errors='coerce') * 100).round() / 100
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        type_col: ['count', (lambda x: x.mode().iloc[0] if len(x.dropna())>0 else 'Unknown')]
    }).reset_index()
    agg.columns = ['lat', 'lon', 'crime_count', 'most_common']
    # compute
    location_cache = {}
    for _, r in agg.iterrows():
        key = f"{r.lat:.4f},{r.lon:.4f}"
        location_cache[key] = {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common)
        }

# ---------- DATA LOADING ----------
def safe_loads():
    global model, encoders, crime_mapping, place_lookup_df, crime_df
    # load model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info("✅ Model loaded from %s", MODEL_PATH)
        else:
            logger.warning("⚠️ Model file not found at %s — continuing without model.", MODEL_PATH)
            model = None
    except Exception as e:
        logger.exception("❌ Failed to load model: %s", e)
        model = None

    # encoders
    try:
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            logger.info("✅ Encoders loaded: %s", list(encoders.keys()))
        else:
            logger.warning("⚠️ Encoders file not found at %s", ENCODER_PATH)
    except Exception as e:
        logger.exception("❌ Failed to load encoders: %s", e)
        encoders = {}

    # crime mapping (index->label)
    try:
        if os.path.exists(CRIME_MAP_PATH):
            with open(CRIME_MAP_PATH, 'r', encoding='utf-8') as f:
                crime_mapping = json.load(f)
            logger.info("✅ Crime mapping loaded (%d entries).", len(crime_mapping))
        else:
            logger.warning("⚠️ Crime mapping not found at %s", CRIME_MAP_PATH)
            crime_mapping = {}
    except Exception as e:
        logger.exception("❌ Failed to load crime mapping: %s", e)
        crime_mapping = {}

    # place lookup
    try:
        if os.path.exists(PLACE_LOOKUP_PATH):
            dfp = pd.read_csv(PLACE_LOOKUP_PATH)
            # normalize columns: try to make columns "Place","Latitude","Longitude"
            cols = {c: c.strip() for c in dfp.columns}
            dfp.rename(columns=cols, inplace=True)
            # find name column
            name_col = None
            for c in dfp.columns:
                low = c.lower()
                if 'place' in low or 'police' in low or 'station' in low or 'name' in low:
                    name_col = c
                    break
            lat_col = next((c for c in dfp.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in dfp.columns if 'lon' in c.lower()), None)
            if name_col is None or lat_col is None or lon_col is None:
                # fallback: try first three columns
                if len(dfp.columns) >= 3:
                    name_col, lat_col, lon_col = dfp.columns[:3]
                else:
                    raise ValueError("place_lookup.csv doesn't have at least 3 usable columns")
            dfp = dfp.rename(columns={name_col: "Place", lat_col: "Latitude", lon_col: "Longitude"})
            # ensure numeric lat/lon
            dfp['Latitude'] = pd.to_numeric(dfp['Latitude'], errors='coerce')
            dfp['Longitude'] = pd.to_numeric(dfp['Longitude'], errors='coerce')
            dfp = dfp.dropna(subset=['Latitude', 'Longitude'])
            place_lookup_df = dfp[['Place', 'Latitude', 'Longitude']].drop_duplicates().reset_index(drop=True)
            logger.info("✅ Loaded %d places from lookup.", len(place_lookup_df))
        else:
            logger.warning("⚠️ place_lookup.csv not found at %s", PLACE_LOOKUP_PATH)
            place_lookup_df = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])
    except Exception as e:
        logger.exception("❌ Failed to load place lookup: %s", e)
        place_lookup_df = pd.DataFrame(columns=["Place", "Latitude", "Longitude"])

    # crime csv
    try:
        global location_cache  # ADD THIS LINE
        if os.path.exists(CRIME_CSV_PATH):
            dfc = pd.read_csv(CRIME_CSV_PATH)
            # normalize expected columns
            # common names may be: Police Station, Year, Type, Date, Time, Place, Latitude, Longitude, Date_fixed
            lat_col = next((c for c in dfc.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in dfc.columns if 'lon' in c.lower()), None)
            if lat_col:
                dfc[lat_col] = pd.to_numeric(dfc[lat_col], errors='coerce')
            if lon_col:
                dfc[lon_col] = pd.to_numeric(dfc[lon_col], errors='coerce')
            dfc = dfc.dropna(subset=[c for c in [lat_col, lon_col] if c is not None])
            crime_df = dfc
            build_cache(crime_df)
            logger.info("✅ Crime data loaded: %d records, cache contains %d clusters.", len(crime_df), len(location_cache))

        else:
            logger.warning("⚠️ Crime CSV not found at %s", CRIME_CSV_PATH)
            crime_df = None
            location_cache = {}
    except Exception as e:
        logger.exception("❌ Failed to load crime data: %s", e)
        crime_df = None
        location_cache = {}

# run on startup
@app.on_event("startup")
def startup_event():
    safe_loads()

# ---------- MODELS FOR REQUEST BODIES ----------
class PredictBody(BaseModel):
    latitude: Optional[float]
    longitude: Optional[float]
    place: Optional[str]
    time: Optional[str]  # "HH:MM" or ISO-ish; we parse "HH:MM"
class SafeRouteBody(BaseModel):
    src: Any
    dst: Any

# ---------- API ROUTES ----------
@app.get("/api")
def api_root():
    return {"status": "ok", "app": APP_TITLE}

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "places": len(place_lookup_df) if place_lookup_df is not None else 0,
        "crime_records": len(crime_df) if crime_df is not None else 0,
        "cached_clusters": len(location_cache),
        "now": datetime.now().isoformat()
    }

@app.get("/api/place-list")
@app.get("/api/place-lookup")  # keep both for compatibility
def place_list():
    try:
        if place_lookup_df is None or place_lookup_df.empty:
            return []
        return place_lookup_df["Place"].dropna().drop_duplicates().sort_values().tolist()
    except Exception as e:
        logger.exception("place_list error: %s", e)
        raise HTTPException(status_code=500, detail="Failed to read place lookup")

@app.get("/api/heatmap")
def heatmap(limit: int = Query(500, ge=1, le=2000)):
    # return heatmap points (lat, lon, risk_score, crime_count, most_common)
    if not location_cache:
        return {"heatmap": [], "total": 0}
    data = []
    for loc in list(location_cache.values())[:limit]:
        risk = calculate_risk_score(loc['lat'], loc['lon'])
        data.append({
            "lat": loc['lat'],
            "lon": loc['lon'],
            "risk_score": risk,
            "crime_count": loc['crime_count'],
            "most_common": loc['most_common']
        })
    data.sort(key=lambda x: x['risk_score'], reverse=True)
    return {"heatmap": data, "total": len(data), "timestamp": datetime.now().isoformat()}

@app.get("/api/heatmap-zones")
def heatmap_zones(limit: int = Query(500, ge=1, le=2000)):
    # zone radius scaled from crime_count -> radius_km
    if not location_cache:
        return {"zones": [], "total": 0}
    zones = []
    for loc in list(location_cache.values())[:limit]:
        crime_count = loc['crime_count']
        # radius: small base + scale, cap at 3 km
        radius_km = min(3.0, max(0.2, (crime_count ** 0.5) / 3.0))
        zones.append({
            "lat": loc['lat'],
            "lon": loc['lon'],
            "crime_count": crime_count,
            "most_common": loc['most_common'],
            "radius_km": round(radius_km, 3),
            "risk_score": calculate_risk_score(loc['lat'], loc['lon'])
        })
    zones.sort(key=lambda x: x['risk_score'], reverse=True)
    return {"zones": zones, "total": len(zones)}

@app.get("/api/predict")
def predict_get(
    place: Optional[str] = Query(None),
    time: Optional[str] = Query(None),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None)
):
    # allow GET style calls from frontend with query params
    return _predict_internal(place=place, time=time, lat=lat, lon=lon)

@app.post("/api/predict")
def predict_post(body: PredictBody):
    return _predict_internal(place=body.place, time=body.time, lat=body.latitude, lon=body.longitude)

def _predict_internal(place: Optional[str], time: Optional[str], lat: Optional[float], lon: Optional[float]):
    if not place and not (lat and lon):
        raise HTTPException(status_code=400, detail="Provide either (lat & lon) or place")

    # resolve place if lat/lon provided but not place
    query_place = place
    if not query_place and lat is not None and lon is not None and not place_lookup_df.empty:
        # nearest by squared distance
        try:
            d2 = (place_lookup_df["Latitude"] - lat)**2 + (place_lookup_df["Longitude"] - lon)**2
            nearest_idx = int(d2.idxmin())
            query_place = place_lookup_df.iloc[nearest_idx]["Place"]
        except Exception:
            query_place = None

    # parse time
    now = datetime.now()
    if time:
        try:
            hour = datetime.strptime(time.strip(), "%H:%M").hour
        except Exception:
            # try just hour "HH"
            try:
                hour = int(time.strip().split(":")[0])
            except Exception:
                hour = now.hour
    else:
        hour = now.hour
    dow = now.weekday()

    # default lat/lon from place if required
    if (lat is None or lon is None) and query_place:
        row = place_lookup_df.loc[place_lookup_df["Place"].str.lower() == str(query_place).lower()]
        if not row.empty:
            lat = float(row.iloc[0]["Latitude"])
            lon = float(row.iloc[0]["Longitude"])

    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Could not resolve latitude/longitude for prediction")

    # attempt model prediction if available
    predicted_crime = None
    confidence = None
    used_model = False
    try:
        if model is not None:
            # build a minimal dataframe consistent with saved encoders if present
            X = {}
            X["Latitude"] = float(lat)
            X["Longitude"] = float(lon)
            X["Hour"] = int(hour)
            X["Month"] = now.month
            # encode PartOfDay, DayOfWeek, Place if encoders exist
            part = ("Morning" if 5 <= hour < 12 else
                    "Afternoon" if 12 <= hour < 17 else
                    "Evening" if 17 <= hour < 21 else "Night")
            if isinstance(encoders, dict):
                # safe transforms
                try:
                    if "PartOfDay" in encoders:
                        X["PartOfDay"] = int(encoders["PartOfDay"].transform([part])[0])
                    if "DayOfWeek" in encoders:
                        X["DayOfWeek"] = int(encoders["DayOfWeek"].transform([now.strftime("%A")])[0])
                    if "Place" in encoders and query_place:
                        X["Place"] = int(encoders["Place"].transform([query_place])[0])
                except Exception:
                    # give up on fancy encoders for this request
                    pass
            X_df = pd.DataFrame([X])
            # model predict_proba
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_df)[0]
                idx = int(np.argmax(probs))
                predicted_crime = crime_mapping.get(str(idx), str(idx)) if crime_mapping else str(idx)
                confidence = round(float(np.max(probs)) * 100, 2)
            else:
                pred = model.predict(X_df)[0]
                predicted_crime = str(pred)
                confidence = None
            used_model = True
    except Exception as e:
        logger.exception("Model prediction failed: %s", e)
        # fall back to historical below

    # historical fallback using nearest cluster
    nearest, dist = None, None
    try:
        nearest, dist = None, None
        min_d = float("inf")
        for v in location_cache.values():
            d = haversine((lat, lon), (v['lat'], v['lon']))
            if d < min_d:
                min_d, nearest = d, v
        dist = min_d if min_d != float("inf") else None
    except Exception:
        pass

    if not predicted_crime and nearest:
        predicted_crime = nearest.get("most_common", "Unknown")
        confidence = max(40, 100 - (dist or 0) * 15) if confidence is None else confidence
    elif not predicted_crime:
        predicted_crime = "Insufficient Data"
        confidence = confidence or 30

    risk_score = calculate_risk_score(lat, lon, hour, dow)
    recs = []
    if risk_score >= 70:
        recs = ["⚠️ HIGH RISK AREA - Avoid if possible", "🚨 Travel in groups", "📱 Share live location"]
    elif risk_score >= 50:
        recs = ["⚡ MODERATE RISK - Exercise caution", "👥 Prefer well-lit, populated routes"]
    else:
        recs = ["✅ RELATIVELY SAFE AREA", "👁️ Stay aware"]

    return {
        "place": query_place,
        "latitude": lat,
        "longitude": lon,
        "predicted_crime": predicted_crime,
        "confidence": confidence,
        "risk_score": risk_score,
        "recommendations": recs,
        "used_model": used_model,
        "distance_to_historical_km": round(dist, 3) if dist is not None else None,
        "hour": hour,
        "day_of_week": dow
    }

@app.post("/api/safe-route")
def safe_route(body: SafeRouteBody):
    # src/dst can be {"lat":..,"lon":..} or place name string
    def resolve(p):
        if isinstance(p, dict) and "lat" in p and "lon" in p:
            return float(p["lat"]), float(p["lon"])
        if isinstance(p, str) and not p.strip().isdigit():
            row = place_lookup_df[place_lookup_df["Place"].str.lower() == p.lower()]
            if not row.empty:
                return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
        raise HTTPException(status_code=400, detail=f"Cannot resolve location: {p}")

    try:
        s = resolve(body.src)
        d = resolve(body.dst)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("resolve error: %s", e)
        raise HTTPException(status_code=400, detail="Invalid src/dst")

    # call OSRM
    try:
        osrm_url = f"https://router.project-osrm.org/route/v1/driving/{s[1]},{s[0]};{d[1]},{d[0]}?overview=full&geometries=geojson"
        resp = requests.get(osrm_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "routes" in data and data["routes"]:
            route = data["routes"][0]
            coords = route["geometry"]["coordinates"]
            # sample up to 60 points for risk along route
            step = max(1, len(coords) // 60)
            samples = [{"lat": c[1], "lon": c[0], "risk": calculate_risk_score(c[1], c[0])} for c in coords[::step]]
            avg_risk = round(sum(p["risk"] for p in samples) / len(samples), 2) if samples else None
            return {
                "geometry": route["geometry"],
                "distance_km": round(route.get("distance", 0) / 1000, 3),
                "duration_min": round(route.get("duration", 0) / 60, 2),
                "average_risk": avg_risk,
                "samples": samples
            }
        else:
            # fallback straight-line sampling
            raise RuntimeError("OSRM returned no routes")
    except Exception as e:
        logger.warning("OSRM failed, using fallback route: %s", e)
        # fallback: straight line points
        n = 30
        pts = []
        for i in range(n + 1):
            t = i / n
            latp = s[0] + t * (d[0] - s[0])
            lonp = s[1] + t * (d[1] - s[1])
            pts.append({"lat": latp, "lon": lonp, "risk": calculate_risk_score(latp, lonp)})
        return {
            "route_points": pts,
            "distance_km": round(haversine(s, d), 3),
            "average_risk": round(sum(p["risk"] for p in pts) / len(pts), 2)
        }

@app.get("/api/dashboard")
def dashboard():
    total = len(crime_df) if crime_df is not None else 0
    heat = heatmap(limit=100)['heatmap'] if location_cache else []
    top_risky = heat[:5]
    recent_30d = 0
    if crime_df is not None:
        date_cols = [c for c in crime_df.columns if 'date' in c.lower()]
        if date_cols:
            try:
                df = crime_df.copy()
                # try to parse a datetime column if present
                if 'Date_fixed' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date_fixed'], errors='coerce')
                elif 'Date' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date'], errors='coerce')
                else:
                    df['__dt'] = None
                recent_30d = int(df[df['__dt'] >= (datetime.now() - timedelta(days=30))].shape[0]) if df['__dt'].notna().any() else 0
            except Exception:
                recent_30d = 0
    avg_risk = round(np.mean([h['risk_score'] for h in heat]) if heat else 0, 2)
    return {
        "total_crimes": total,
        "recent_crimes_30d": recent_30d,
        "top_risky": top_risky,
        "average_risk": avg_risk,
        "cached_clusters": len(location_cache)
    }

@app.get("/api/crime-trends")
def crime_trends():
    if crime_df is None:
        return {"monthly_trends": [], "crime_types": {}, "hourly_distribution": {}}
    df = crime_df.copy()
    # try to build datetime
    if 'Date_fixed' in df.columns:
        df['__dt'] = pd.to_datetime(df['Date_fixed'], errors='coerce')
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['__dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    else:
        df['__dt'] = None
    monthly = []
    try:
        if df['__dt'].notna().any():
            m = df.dropna(subset=['__dt']).groupby([df['__dt'].dt.year, df['__dt'].dt.month]).size().reset_index(name='count')
            m['date'] = pd.to_datetime(m[[m.columns[0], m.columns[1]]].assign(day=1))
            m = m.sort_values('date').tail(12)
            for _, r in m.iterrows():
                monthly.append({"date": r['date'].strftime("%Y-%m"), "count": int(r['count'])})
    except Exception:
        monthly = []
    # crime types
    type_col = next((c for c in df.columns if c.lower() in ('type', 'crime')), None)
    crime_types = df[type_col].value_counts().head(10).to_dict() if type_col and type_col in df.columns else {}
    # hourly distribution
    hour_col = next((c for c in df.columns if 'hour' in c.lower()), None)
    if hour_col and hour_col in df.columns:
        hourly = df[hour_col].value_counts().sort_index().to_dict()
    else:
        # try deriving from __dt
        if df['__dt'].notna().any():
            hourly = df.dropna(subset=['__dt'])['__dt'].dt.hour.value_counts().sort_index().to_dict()
        else:
            hourly = {}
    return {"monthly_trends": monthly, "crime_types": {k: int(v) for k, v in crime_types.items()}, "hourly_distribution": {int(k): int(v) for k, v in hourly.items()}}

# ---------- RUN ----------
if __name__ == "__main__":
    import uvicorn
    safe_loads()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
