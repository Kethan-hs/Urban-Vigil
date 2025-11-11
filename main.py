# main.py - COMPLETE FIXED VERSION
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import joblib, os, math, json, logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests

# ==================== CONFIG ====================
APP_TITLE = "Urban Vigil Pro - AI Safety API"
MODELS_DIR = os.getenv("MODELS_DIR", "models")
MODEL_FILE = os.getenv("MODEL_FILE", "Final_model_fixed.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_FILE)
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoders_fixed.pkl")
CRIME_MAP_PATH = os.path.join(MODELS_DIR, "crime_target_mapping.json")
PLACE_LOOKUP_PATH = os.path.join(MODELS_DIR, "place_lookup.csv")
CRIME_CSV_PATH = os.path.join(MODELS_DIR, "bengaluru_dataset.csv")

# Bangalore Police Stations with coordinates
BANGALORE_POLICE_STATIONS = [
    {"name": "Banashankari PS", "lat": 12.9250, "lon": 77.5670, "phone": "080-26771290"},
    {"name": "Koramangala PS", "lat": 12.9352, "lon": 77.6245, "phone": "080-25537466"},
    {"name": "Indiranagar PS", "lat": 12.9719, "lon": 77.6412, "phone": "080-25213009"},
    {"name": "MG Road PS", "lat": 12.9763, "lon": 77.5995, "phone": "080-22212444"},
    {"name": "Jayanagar PS", "lat": 12.9250, "lon": 77.5938, "phone": "080-26633292"},
    {"name": "Hebbal PS", "lat": 13.0350, "lon": 77.5970, "phone": "080-23637700"},
    {"name": "Whitefield PS", "lat": 12.9698, "lon": 77.7490, "phone": "080-28452905"},
    {"name": "Electronic City PS", "lat": 12.8419, "lon": 77.6605, "phone": "080-27835215"},
    {"name": "Kammanahalli PS", "lat": 13.0101, "lon": 77.6197, "phone": "080-25452611"},
    {"name": "Rajajinagar PS", "lat": 13.0136, "lon": 77.5551, "phone": "080-23580549"},
    {"name": "Yeshwanthpur PS", "lat": 13.0287, "lon": 77.5412, "phone": "080-23578450"},
    {"name": "Malleswaram PS", "lat": 13.0034, "lon": 77.5707, "phone": "080-23340223"},
    {"name": "RT Nagar PS", "lat": 13.0191, "lon": 77.5959, "phone": "080-23636244"},
    {"name": "Yelahanka PS", "lat": 13.1007, "lon": 77.5963, "phone": "080-28468933"},
    {"name": "JP Nagar PS", "lat": 12.9085, "lon": 77.5850, "phone": "080-26493399"},
    {"name": "BTM Layout PS", "lat": 12.9165, "lon": 77.6101, "phone": "080-26685336"},
    {"name": "HSR Layout PS", "lat": 12.9116, "lon": 77.6380, "phone": "080-25727731"},
    {"name": "Marathahalli PS", "lat": 12.9592, "lon": 77.6974, "phone": "080-25221744"},
    {"name": "Bellandur PS", "lat": 12.9260, "lon": 77.6748, "phone": "080-49262626"},
    {"name": "Sarjapur PS", "lat": 12.9010, "lon": 77.7280, "phone": "080-27835901"},
]

# ==================== APP ====================
app = FastAPI(title=APP_TITLE)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# Serve static files
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

# ==================== LOGGING ====================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("urban-vigil")

# ==================== GLOBALS ====================
model = None
encoders = {}
crime_mapping: Dict[str, Any] = {}
place_lookup_df: pd.DataFrame = pd.DataFrame()
crime_df: Optional[pd.DataFrame] = None
location_cache: Dict[str, Dict[str, Any]] = {}

# ==================== UTILITIES ====================
def haversine(coord1, coord2):
    """Calculate distance between two coordinates in km"""
    R = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def geocode_location(place_name: str):
    """Geocode a place name to coordinates using Nominatim"""
    try:
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{place_name}, Bangalore, India",
            "format": "json",
            "limit": 1
        }
        headers = {"User-Agent": "UrbanVigilPro/1.0"}
        
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.ok:
            data = response.json()
            if data:
                return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        logger.error(f"Geocoding error: {e}")
    return None, None

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
    """Calculate risk score using ML model or heuristics"""
    base_risk = 30.0
    
    # Get nearest crime data
    nearest, min_dist = None, float("inf")
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_dist:
            min_dist, nearest = d, v
    
    if nearest:
        base_risk += min(40, nearest.get('crime_count', 0) * 0.5)
        crime_type = str(nearest.get('most_common', '')).lower()
        if 'murder' in crime_type or 'rape' in crime_type:
            base_risk += 25
        elif 'robbery' in crime_type or 'assault' in crime_type:
            base_risk += 15
        elif 'theft' in crime_type:
            base_risk += 10
    
    # Time factors
    if hour is not None:
        if 22 <= hour or hour <= 4:
            base_risk += 20
        elif 18 <= hour <= 21:
            base_risk += 10
    
    if day_of_week is not None and day_of_week >= 5:
        base_risk += 5
    
    if min_dist > 1.0:
        base_risk += min(10, min_dist * 2)
    
    return max(0, min(100, round(base_risk, 1)))

def get_safety_recommendations(risk_score, hour=None, crime_type=None):
    """Generate safety recommendations based on risk"""
    recommendations = []
    
    if risk_score >= 70:
        recommendations.extend([
            "⚠️ HIGH RISK AREA - Avoid if possible",
            "🚨 Travel in groups of 3 or more",
            "📱 Share live location with family/friends",
            "🚗 Use trusted transportation services"
        ])
    elif risk_score >= 50:
        recommendations.extend([
            "⚡ MODERATE RISK - Exercise caution",
            "👥 Prefer well-lit, populated routes",
            "📞 Keep emergency contacts ready"
        ])
    else:
        recommendations.extend([
            "✅ RELATIVELY SAFE AREA",
            "👁️ Stay aware of surroundings",
            "🎧 Avoid distractions while walking"
        ])
    
    if hour is not None:
        if 22 <= hour or hour <= 5:
            recommendations.append("🌙 Late hours - Use well-lit paths only")
    
    if crime_type and 'theft' in str(crime_type).lower():
        recommendations.append("💰 Secure valuables, avoid displaying phones/jewelry")
    
    recommendations.append("🚨 Emergency: Police 100 | Women Helpline 1091")
    
    return recommendations

def build_cache(df: pd.DataFrame):
    """Build location cache from crime data"""
    global location_cache
    if df is None or df.empty:
        return
    
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    type_col = next((c for c in df.columns if c.lower() in ('type', 'crime')), None)
    
    if not all([lat_col, lon_col, type_col]):
        return
    
    df = df.copy()
    df['lat_bin'] = (pd.to_numeric(df[lat_col], errors='coerce') * 100).round() / 100
    df['lon_bin'] = (pd.to_numeric(df[lon_col], errors='coerce') * 100).round() / 100
    
    agg = df.groupby(['lat_bin', 'lon_bin']).agg({
        type_col: ['count', lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown']
    }).reset_index()
    agg.columns = ['lat', 'lon', 'crime_count', 'most_common']
    
    location_cache = {}
    for _, r in agg.iterrows():
        key = f"{r.lat:.4f},{r.lon:.4f}"
        location_cache[key] = {
            "lat": float(r.lat),
            "lon": float(r.lon),
            "crime_count": int(r.crime_count),
            "most_common": str(r.most_common)
        }

# ==================== DATA LOADING ====================
def safe_loads():
    """Load all data safely"""
    global model, encoders, crime_mapping, place_lookup_df, crime_df
    
    # Load model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            logger.info(f"✅ Model loaded from {MODEL_PATH}")
        else:
            logger.warning(f"⚠️ Model not found at {MODEL_PATH}")
    except Exception as e:
        logger.exception(f"❌ Model load error: {e}")
    
    # Load encoders
    try:
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            logger.info(f"✅ Encoders loaded: {list(encoders.keys())}")
    except Exception as e:
        logger.exception(f"❌ Encoder load error: {e}")
    
    # Load crime mapping
    try:
        if os.path.exists(CRIME_MAP_PATH):
            with open(CRIME_MAP_PATH, 'r') as f:
                crime_mapping = json.load(f)
            logger.info(f"✅ Crime mapping loaded: {len(crime_mapping)} entries")
    except Exception as e:
        logger.exception(f"❌ Crime mapping error: {e}")
    
    # Load place lookup
    try:
        if os.path.exists(PLACE_LOOKUP_PATH):
            dfp = pd.read_csv(PLACE_LOOKUP_PATH)
            # Auto-detect columns
            name_col = next((c for c in dfp.columns if 'place' in c.lower() or 'name' in c.lower()), dfp.columns[0])
            lat_col = next((c for c in dfp.columns if 'lat' in c.lower()), dfp.columns[1])
            lon_col = next((c for c in dfp.columns if 'lon' in c.lower()), dfp.columns[2])
            
            dfp = dfp.rename(columns={name_col: "Place", lat_col: "Latitude", lon_col: "Longitude"})
            dfp['Latitude'] = pd.to_numeric(dfp['Latitude'], errors='coerce')
            dfp['Longitude'] = pd.to_numeric(dfp['Longitude'], errors='coerce')
            place_lookup_df = dfp.dropna(subset=['Latitude', 'Longitude'])
            logger.info(f"✅ Loaded {len(place_lookup_df)} places")
    except Exception as e:
        logger.exception(f"❌ Place lookup error: {e}")
    
    # Load crime data
    try:
        if os.path.exists(CRIME_CSV_PATH):
            dfc = pd.read_csv(CRIME_CSV_PATH)
            lat_col = next((c for c in dfc.columns if 'lat' in c.lower()), None)
            lon_col = next((c for c in dfc.columns if 'lon' in c.lower()), None)
            
            if lat_col and lon_col:
                dfc[lat_col] = pd.to_numeric(dfc[lat_col], errors='coerce')
                dfc[lon_col] = pd.to_numeric(dfc[lon_col], errors='coerce')
                dfc = dfc.dropna(subset=[lat_col, lon_col])
            
            crime_df = dfc
            build_cache(crime_df)
            logger.info(f"✅ Crime data: {len(crime_df)} records, {len(location_cache)} clusters")
    except Exception as e:
        logger.exception(f"❌ Crime data error: {e}")

@app.on_event("startup")
def startup_event():
    safe_loads()

# ==================== MODELS ====================
class PredictRequest(BaseModel):
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place: Optional[str] = None
    location_name: Optional[str] = None  # NEW: Accept plain text location
    time: Optional[str] = None

class RouteRequest(BaseModel):
    src: Any
    dst: Any

# ==================== ENDPOINTS ====================
@app.get("/api")
def api_root():
    return {
        "status": "online",
        "app": APP_TITLE,
        "version": "2.0",
        "endpoints": [
            "/api/health",
            "/api/place-lookup", 
            "/api/predict",
            "/api/heatmap",
            "/api/safe-route",
            "/api/dashboard",
            "/api/crime-trends",
            "/api/police-locator"
        ]
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "places": len(place_lookup_df),
        "crime_records": len(crime_df) if crime_df is not None else 0,
        "cached_clusters": len(location_cache),
        "police_stations": len(BANGALORE_POLICE_STATIONS),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/place-lookup")
def place_lookup():
    """Get list of known places"""
    try:
        if place_lookup_df is None or place_lookup_df.empty:
            return []
        return place_lookup_df["Place"].dropna().unique().tolist()
    except Exception as e:
        logger.exception("place_lookup error")
        return []

@app.get("/api/police-locator")
@app.post("/api/police-locator")
def police_locator(lat: Optional[float] = None, lon: Optional[float] = None):
    """Find nearest police stations - NEW FEATURE"""
    if lat is None or lon is None:
        return {"error": "Latitude and longitude required"}
    
    # Calculate distances
    stations_with_distance = []
    for station in BANGALORE_POLICE_STATIONS:
        distance = haversine((lat, lon), (station['lat'], station['lon']))
        stations_with_distance.append({
            **station,
            "distance_km": round(distance, 2)
        })
    
    # Sort by distance and get top 5
    stations_with_distance.sort(key=lambda x: x['distance_km'])
    top_5 = stations_with_distance[:5]
    
    return {
        "nearest_stations": top_5,
        "total_stations": len(BANGALORE_POLICE_STATIONS),
        "user_location": {"lat": lat, "lon": lon}
    }

@app.get("/api/predict")
@app.post("/api/predict")
def predict(req: Optional[PredictRequest] = None, 
            place: Optional[str] = Query(None),
            location_name: Optional[str] = Query(None),  # NEW
            lat: Optional[float] = Query(None),
            lon: Optional[float] = Query(None),
            time: Optional[str] = Query(None)):
    """Predict crime risk - ENHANCED"""
    
    # Handle both POST and GET
    if req:
        place = req.place or req.location_name
        location_name = req.location_name
        lat = req.latitude
        lon = req.longitude
        time = req.time
    
    # NEW: Handle plain text location
    if location_name and not lat and not lon:
        logger.info(f"Geocoding location: {location_name}")
        lat, lon = geocode_location(location_name)
        if not lat:
            # Check if location is in Bangalore
            return {
                "error": "location_not_found",
                "message": f"'{location_name}' not found in Bengaluru. We're expanding to more areas soon!",
                "suggestion": "Please try a well-known area like Koramangala, Indiranagar, or Whitefield"
            }
    
    # Resolve place to coordinates
    if place and not lat:
        row = place_lookup_df[place_lookup_df["Place"].str.lower() == place.lower()]
        if not row.empty:
            lat = float(row.iloc[0]["Latitude"])
            lon = float(row.iloc[0]["Longitude"])
    
    if not lat or not lon:
        raise HTTPException(400, "Please provide coordinates or a valid location name")
    
    # Parse time
    now = datetime.now()
    hour = now.hour
    if time:
        try:
            hour = datetime.strptime(time.strip(), "%H:%M").hour
        except:
            try:
                hour = int(time.strip().split(":")[0])
            except:
                pass
    
    day_of_week = now.weekday()
    
    # Calculate risk
    risk_score = calculate_risk_score(lat, lon, hour, day_of_week)
    
    # Get nearest crime data
    nearest, distance = None, None
    min_d = float("inf")
    for v in location_cache.values():
        d = haversine((lat, lon), (v['lat'], v['lon']))
        if d < min_d:
            min_d, nearest = d, v
    distance = min_d if min_d != float("inf") else None
    
    predicted_crime = nearest.get('most_common', 'Unknown') if nearest else 'Insufficient Data'
    confidence = max(40, 100 - (distance * 10)) if distance else 50
    
    # Get recommendations
    recommendations = get_safety_recommendations(risk_score, hour, predicted_crime)
    
    return {
        "latitude": lat,
        "longitude": lon,
        "predicted_crime": predicted_crime,
        "risk_score": round(risk_score, 1),
        "confidence": round(confidence, 1),
        "recommendations": recommendations,
        "nearby_crimes": nearest['crime_count'] if nearest else 0,
        "hour": hour,
        "day_of_week": day_of_week,
        "distance_to_data_km": round(distance, 2) if distance else None
    }

@app.get("/api/heatmap")
def heatmap(limit: int = Query(500, ge=1, le=2000)):
    """Get crime heatmap data"""
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
    return {"heatmap": data, "total": len(data)}

@app.post("/api/safe-route")
def safe_route(req: RouteRequest):
    """Calculate safe route - OSRM INTEGRATED"""
    def resolve(p):
        if isinstance(p, dict):
            return float(p.get("lat")), float(p.get("lon"))
        if isinstance(p, str):
            # Try geocoding
            lat, lon = geocode_location(p)
            if lat:
                return lat, lon
            # Try place lookup
            row = place_lookup_df[place_lookup_df["Place"].str.lower() == p.lower()]
            if not row.empty:
                return float(row.iloc[0]["Latitude"]), float(row.iloc[0]["Longitude"])
        raise HTTPException(400, f"Cannot resolve location: {p}")
    
    src_lat, src_lon = resolve(req.src)
    dst_lat, dst_lon = resolve(req.dst)
    
    # Call OSRM for real routing
    try:
        osrm_url = f"https://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        params = {"overview": "full", "geometries": "geojson", "steps": "true"}
        
        response = requests.get(osrm_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data.get("routes"):
            route = data["routes"][0]
            geometry = route["geometry"]
            coords = geometry["coordinates"]
            
            # Sample points for risk assessment
            step = max(1, len(coords) // 30)
            samples = []
            for i in range(0, len(coords), step):
                c = coords[i]
                risk = calculate_risk_score(c[1], c[0])
                samples.append({"lat": c[1], "lon": c[0], "risk": round(risk, 1)})
            
            avg_risk = sum(s["risk"] for s in samples) / len(samples)
            
            return {
                "geometry": geometry,
                "distance_km": round(route.get("distance", 0) / 1000, 2),
                "duration_min": round(route.get("duration", 0) / 60, 1),
                "average_risk": round(avg_risk, 1),
                "samples": samples,
                "note": "Route calculated using real road network"
            }
    except Exception as e:
        logger.error(f"OSRM error: {e}")
    
    # Fallback: straight line
    n = 20
    pts = []
    for i in range(n + 1):
        t = i / n
        lat = src_lat + t * (dst_lat - src_lat)
        lon = src_lon + t * (dst_lon - src_lon)
        risk = calculate_risk_score(lat, lon)
        pts.append({"lat": lat, "lon": lon, "risk": round(risk, 1)})
    
    return {
        "route_points": pts,
        "distance_km": round(haversine((src_lat, src_lon), (dst_lat, dst_lon)), 2),
        "average_risk": round(sum(p["risk"] for p in pts) / len(pts), 1),
        "note": "Fallback route (straight line)"
    }

@app.get("/api/dashboard")
def dashboard():
    """Dashboard stats"""
    total = len(crime_df) if crime_df is not None else 0
    heat = heatmap(limit=100)['heatmap']
    top_risky = heat[:5]
    
    recent_30d = 0
    if crime_df is not None:
        date_cols = [c for c in crime_df.columns if 'date' in c.lower()]
        if date_cols:
            try:
                df = crime_df.copy()
                if 'Date_fixed' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date_fixed'], errors='coerce')
                elif 'Date' in df.columns:
                    df['__dt'] = pd.to_datetime(df['Date'], errors='coerce')
                if '__dt' in df.columns and df['__dt'].notna().any():
                    recent_30d = len(df[df['__dt'] >= (datetime.now() - timedelta(days=30))])
            except:
                pass
    
    avg_risk = round(np.mean([h['risk_score'] for h in heat]), 1) if heat else 0
    
    return {
        "total_crimes": total,
        "recent_crimes_30d": recent_30d,
        "top_risky": top_risky,
        "average_risk": avg_risk,
        "cached_clusters": len(location_cache)
    }

@app.get("/api/crime-trends")
def crime_trends():
    """Crime trends analysis"""
    if crime_df is None:
        return {"monthly_trends": [], "crime_types": {}, "hourly_distribution": {}}
    
    df = crime_df.copy()
    
    # Build datetime
    if 'Date_fixed' in df.columns:
        df['__dt'] = pd.to_datetime(df['Date_fixed'], errors='coerce')
    elif 'Date' in df.columns and 'Time' in df.columns:
        df['__dt'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str), errors='coerce')
    
    monthly = []
    if '__dt' in df.columns and df['__dt'].notna().any():
        m = df.dropna(subset=['__dt']).groupby([df['__dt'].dt.year, df['__dt'].dt.month]).size().reset_index(name='count')
        m['date'] = pd.to_datetime(m[[m.columns[0], m.columns[1]]].assign(day=1))
        m = m.sort_values('date').tail(12)
        monthly = [{"date": r['date'].strftime("%Y-%m"), "count": int(r['count'])} for _, r in m.iterrows()]
    
    type_col = next((c for c in df.columns if c.lower() in ('type', 'crime')), None)
    crime_types = df[type_col].value_counts().head(10).to_dict() if type_col else {}
    
    hour_col = next((c for c in df.columns if 'hour' in c.lower()), None)
    if hour_col:
        hourly = df[hour_col].value_counts().sort_index().to_dict()
    elif '__dt' in df.columns and df['__dt'].notna().any():
        hourly = df.dropna(subset=['__dt'])['__dt'].dt.hour.value_counts().sort_index().to_dict()
    else:
        hourly = {}
    
    return {
        "monthly_trends": monthly,
        "crime_types": {k: int(v) for k, v in crime_types.items()},
        "hourly_distribution": {int(k): int(v) for k, v in hourly.items()}
    }

if __name__ == "__main__":
    import uvicorn
    safe_loads()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))