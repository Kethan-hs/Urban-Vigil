from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import defaultdict
import joblib, os, math, json, hashlib
import numpy as np
import pandas as pd

# ==================== CONFIG ====================
APP_TITLE = "Urban Vigil Pro - AI-Powered Safety Platform"
MODEL_PATH = os.getenv("MODEL_PATH", "crime_predictor_bangalore_new.pkl")
DATA_PATH = os.getenv("DATA_PATH", "bengaluru_dataset.csv")  # Your dataset
CACHE_DURATION = 300  # 5 minutes cache

app = FastAPI(title=APP_TITLE, version="2.0")

# CORS - Allow all for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL STATE ====================
model = None
model_info = {"loaded": False, "type": None, "features": None}
crime_data = None
location_cache = {}
risk_cache = {}

# Bengaluru Safety Contacts
SAFETY_CONTACTS = {
    "emergency": [
        {"name": "Police Emergency", "number": "100", "type": "police"},
        {"name": "National Emergency", "number": "112", "type": "emergency"},
        {"name": "Women Helpline", "number": "1091", "type": "women"},
        {"name": "Ambulance", "number": "108", "type": "medical"},
        {"name": "Fire", "number": "101", "type": "fire"},
    ],
    "police_stations": [
        {"name": "Banashankari Police Station", "number": "080-26771290", "area": "Banashankari"},
        {"name": "Koramangala Police Station", "number": "080-25537466", "area": "Koramangala"},
        {"name": "Indiranagar Police Station", "number": "080-25213009", "area": "Indiranagar"},
        {"name": "Whitefield Police Station", "number": "080-28452905", "area": "Whitefield"},
        {"name": "Jayanagar Police Station", "number": "080-26633292", "area": "Jayanagar"},
    ]
}

# ==================== DATA LOADING ====================
def load_model():
    """Load the trained model safely"""
    global model, model_info
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model not found at {MODEL_PATH}")
        return False
    
    try:
        model = joblib.load(MODEL_PATH)
        model_info = {
            "loaded": True,
            "type": type(model).__name__,
            "features": getattr(model, "feature_names_in_", None),
            "n_features": getattr(model, "n_features_in_", None),
        }
        print(f"✅ Model loaded: {model_info['type']}")
        return True
    except Exception as e:
        print(f"❌ Model load error: {e}")
        return False

def load_crime_data():
    """Load and process crime dataset"""
    global crime_data, location_cache
    if not os.path.exists(DATA_PATH):
        print(f"⚠️  Dataset not found at {DATA_PATH}")
        return False
    
    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)
        print(f"📊 Loaded {len(df)} crime records")
        
        # Process columns
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Parse datetime
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
        
        # Clean coordinates
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        crime_data = df
        
        # Build location cache (aggregate by area)
        build_location_cache()
        
        print(f"✅ Processed crime data: {len(crime_data)} valid records")
        return True
    except Exception as e:
        print(f"❌ Data load error: {e}")
        return False

def build_location_cache():
    """Build aggregated crime statistics by location"""
    global location_cache
    
    if crime_data is None:
        return
    
    # Group by approximate location (0.01 degree ~ 1km grid)
    crime_data['lat_bin'] = (crime_data['latitude'] * 100).round() / 100
    crime_data['lon_bin'] = (crime_data['longitude'] * 100).round() / 100
    
    grouped = crime_data.groupby(['lat_bin', 'lon_bin']).agg({
        'type': ['count', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'],
        'hour': 'mean',
        'police_station': lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'
    }).reset_index()
    
    grouped.columns = ['lat', 'lon', 'crime_count', 'most_common_crime', 'avg_hour', 'police_station']
    
    for _, row in grouped.iterrows():
        key = f"{row['lat']:.2f},{row['lon']:.2f}"
        location_cache[key] = {
            "lat": float(row['lat']),
            "lon": float(row['lon']),
            "crime_count": int(row['crime_count']),
            "most_common": str(row['most_common_crime']),
            "avg_hour": float(row['avg_hour']),
            "police_station": str(row['police_station'])
        }
    
    print(f"📍 Built cache for {len(location_cache)} locations")

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_model()
    load_crime_data()

# ==================== HELPER FUNCTIONS ====================
def haversine(coord1, coord2):
    """Calculate distance between two coordinates in km"""
    R = 6371.0
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def get_nearest_location_data(lat, lon, radius_km=2.0):
    """Get crime statistics for nearest location"""
    nearest = None
    min_dist = float('inf')
    
    for loc_data in location_cache.values():
        dist = haversine((lat, lon), (loc_data['lat'], loc_data['lon']))
        if dist < min_dist and dist <= radius_km:
            min_dist = dist
            nearest = loc_data
    
    return nearest, min_dist

def calculate_risk_score(lat, lon, hour=None, day_of_week=None):
    """Calculate risk score using historical data + heuristics"""
    # Get historical data
    location_data, distance = get_nearest_location_data(lat, lon)
    
    base_risk = 30  # Baseline
    
    if location_data:
        # Crime density factor (0-40 points)
        crime_count = location_data['crime_count']
        density_score = min(40, crime_count * 2)  # Cap at 40
        base_risk += density_score
        
        # Most common crime severity
        crime_type = location_data.get('most_common', '').lower()
        if 'murder' in crime_type or 'rape' in crime_type:
            base_risk += 25
        elif 'robbery' in crime_type or 'assault' in crime_type:
            base_risk += 15
        elif 'theft' in crime_type or 'burglary' in crime_type:
            base_risk += 10
    
    # Time-based factors
    if hour is not None:
        if 22 <= hour or hour <= 4:  # Late night/early morning
            base_risk += 20
        elif 18 <= hour <= 21:  # Evening
            base_risk += 10
        elif 6 <= hour <= 9:  # Morning rush
            base_risk += 5
    
    # Day of week
    if day_of_week is not None and day_of_week >= 5:  # Weekend
        base_risk += 5
    
    # Distance penalty (data is far)
    if distance and distance > 1.0:
        base_risk += min(10, distance * 2)
    
    return min(100, max(0, base_risk))

def get_safety_recommendations(risk_score, hour=None, crime_type=None):
    """Generate contextual safety recommendations"""
    recommendations = []
    
    if risk_score >= 70:
        recommendations.append("⚠️ HIGH RISK AREA - Avoid if possible")
        recommendations.append("🚨 Travel in groups, stay alert")
        recommendations.append("📱 Share live location with trusted contacts")
    elif risk_score >= 50:
        recommendations.append("⚡ MODERATE RISK - Exercise caution")
        recommendations.append("👥 Prefer well-lit, populated routes")
    else:
        recommendations.append("✅ RELATIVELY SAFE AREA")
        recommendations.append("👁️ Stay aware of surroundings")
    
    # Time-based
    if hour is not None:
        if 22 <= hour or hour <= 5:
            recommendations.append("🌙 Late hours - Use trusted transportation")
        elif 18 <= hour <= 21:
            recommendations.append("🌆 Evening time - Stick to main roads")
    
    # Crime-specific
    if crime_type:
        crime_lower = str(crime_type).lower()
        if 'theft' in crime_lower or 'robbery' in crime_lower:
            recommendations.append("💰 Secure valuables, avoid displaying expensive items")
        if 'assault' in crime_lower:
            recommendations.append("🥊 Stay in well-populated areas")
    
    # General
    recommendations.append("📞 Emergency: 112 | Police: 100 | Women: 1091")
    
    return recommendations

def predict_with_model(lat, lon, hour, day_of_week):
    """Try to predict using the loaded model"""
    if model is None:
        raise ValueError("Model not loaded")
    
    # Prepare features based on your model's requirements
    # This is a guess - adjust based on your actual model
    features = np.array([[lat, lon, hour, day_of_week]])
    
    try:
        # Try prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            pred_class = model.classes_[np.argmax(proba)]
            confidence = float(np.max(proba) * 100)
        else:
            pred_class = model.predict(features)[0]
            confidence = 75.0
        
        return str(pred_class), confidence
    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

# ==================== MODELS ====================
class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-90, le=90)
    hour: Optional[int] = Field(None, ge=0, le=23)
    day_of_week: Optional[int] = Field(None, ge=0, le=6)

class RouteRequest(BaseModel):
    src_lat: float
    src_lon: float
    dst_lat: float
    dst_lon: float
    departure_time: Optional[str] = None

# ==================== ENDPOINTS ====================
@app.get("/")
def root():
    return {
        "status": "online",
        "app": APP_TITLE,
        "version": "2.0",
        "model_loaded": model_info["loaded"],
        "data_loaded": crime_data is not None,
        "total_records": len(crime_data) if crime_data is not None else 0,
        "cached_locations": len(location_cache)
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model": model_info,
        "data_records": len(crime_data) if crime_data is not None else 0,
        "cached_locations": len(location_cache),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict crime risk for a location"""
    lat, lon = req.latitude, req.longitude
    hour = req.hour if req.hour is not None else datetime.now().hour
    day = req.day_of_week if req.day_of_week is not None else datetime.now().weekday()
    
    # Try model prediction first
    predicted_crime = None
    confidence = None
    used_model = False
    
    if model_info["loaded"]:
        try:
            predicted_crime, confidence = predict_with_model(lat, lon, hour, day)
            used_model = True
        except Exception as e:
            print(f"Model prediction failed: {e}")
    
    # Calculate risk score
    risk_score = calculate_risk_score(lat, lon, hour, day)
    
    # Get location context
    location_data, distance = get_nearest_location_data(lat, lon)
    
    # If no model prediction, use historical data
    if not predicted_crime and location_data:
        predicted_crime = location_data.get('most_common', 'Unknown')
        confidence = max(50, 100 - (distance * 10))  # Confidence decreases with distance
    elif not predicted_crime:
        predicted_crime = "Insufficient Data"
        confidence = 30
    
    # Generate recommendations
    recommendations = get_safety_recommendations(risk_score, hour, predicted_crime)
    
    return {
        "latitude": lat,
        "longitude": lon,
        "predicted_crime": predicted_crime,
        "risk_score": round(risk_score, 1),
        "confidence": round(confidence, 1),
        "recommendations": recommendations,
        "nearby_crimes": location_data['crime_count'] if location_data else 0,
        "nearest_police": location_data['police_station'] if location_data else "Unknown",
        "distance_to_data": round(distance, 2) if distance else None,
        "used_model": used_model,
        "hour": hour,
        "day_of_week": day
    }

@app.get("/api/heatmap")
def get_heatmap(limit: int = 100):
    """Get crime heatmap data"""
    if not location_cache:
        return {"heatmap": [], "total": 0}
    
    heatmap_data = []
    
    for loc_data in list(location_cache.values())[:limit]:
        risk = calculate_risk_score(loc_data['lat'], loc_data['lon'])
        heatmap_data.append({
            "lat": loc_data['lat'],
            "lon": loc_data['lon'],
            "risk_score": round(risk, 1),
            "crime_count": loc_data['crime_count'],
            "most_common": loc_data['most_common'],
            "police_station": loc_data['police_station']
        })
    
    # Sort by risk
    heatmap_data.sort(key=lambda x: x['risk_score'], reverse=True)
    
    return {
        "heatmap": heatmap_data,
        "total": len(heatmap_data),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/crime-trends")
def get_crime_trends():
    """Get crime trends over time"""
    if crime_data is None:
        return {"trends": [], "message": "No data available"}
    
    # Monthly trends
    monthly = crime_data.groupby(['year', 'month']).size().reset_index(name='count')
    monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
    monthly = monthly.sort_values('date').tail(12)  # Last 12 months
    
    trends = []
    for _, row in monthly.iterrows():
        trends.append({
            "date": row['date'].strftime('%Y-%m'),
            "count": int(row['count'])
        })
    
    # Crime type distribution
    crime_types = crime_data['type'].value_counts().head(10).to_dict()
    
    # Hourly distribution
    hourly = crime_data['hour'].value_counts().sort_index().to_dict()
    
    return {
        "monthly_trends": trends,
        "crime_types": {k: int(v) for k, v in crime_types.items()},
        "hourly_distribution": {int(k): int(v) for k, v in hourly.items()},
        "total_crimes": len(crime_data),
        "date_range": {
            "start": crime_data['datetime'].min().isoformat() if 'datetime' in crime_data.columns else None,
            "end": crime_data['datetime'].max().isoformat() if 'datetime' in crime_data.columns else None
        }
    }

@app.post("/api/safe-route")
def safe_route(req: RouteRequest):
    """Calculate safest route between two points"""
    src = (req.src_lat, req.src_lon)
    dst = (req.dst_lat, req.dst_lon)
    
    # Calculate risk at source and destination
    src_risk = calculate_risk_score(src[0], src[1])
    dst_risk = calculate_risk_score(dst[0], dst[1])
    
    # Sample points along the direct route
    num_samples = 10
    route_points = []
    total_risk = 0
    
    for i in range(num_samples + 1):
        t = i / num_samples
        lat = src[0] + t * (dst[0] - src[0])
        lon = src[1] + t * (dst[1] - src[1])
        risk = calculate_risk_score(lat, lon)
        
        route_points.append({
            "lat": lat,
            "lon": lon,
            "risk": round(risk, 1)
        })
        total_risk += risk
    
    avg_risk = total_risk / (num_samples + 1)
    distance = haversine(src, dst)
    
    # Generate recommendations
    recommendations = get_safety_recommendations(avg_risk)
    
    return {
        "route": route_points,
        "distance_km": round(distance, 2),
        "average_risk": round(avg_risk, 1),
        "src_risk": round(src_risk, 1),
        "dst_risk": round(dst_risk, 1),
        "recommendations": recommendations,
        "high_risk_segments": sum(1 for p in route_points if p['risk'] >= 70),
        "estimated_time_min": round(distance / 40 * 60, 0)  # Assuming 40km/h avg
    }

@app.get("/api/safety-contacts")
def get_safety_contacts():
    """Get emergency contacts and helplines"""
    return SAFETY_CONTACTS

@app.get("/api/dashboard")
def get_dashboard():
    """Get dashboard statistics"""
    if crime_data is None:
        return {"error": "No data available"}
    
    # Overall stats
    total_crimes = len(crime_data)
    
    # Top risky areas
    heatmap = get_heatmap(limit=10)
    top_risky = heatmap['heatmap'][:5]
    
    # Recent trends
    if 'datetime' in crime_data.columns:
        recent = crime_data[crime_data['datetime'] >= (datetime.now() - timedelta(days=30))]
        recent_count = len(recent)
    else:
        recent_count = 0
    
    # Crime type distribution
    crime_types = crime_data['type'].value_counts().head(5).to_dict()
    
    # Average risk
    avg_risk = np.mean([loc['risk_score'] for loc in heatmap['heatmap']]) if heatmap['heatmap'] else 0
    
    return {
        "total_crimes": total_crimes,
        "recent_crimes_30d": recent_count,
        "top_risky_areas": top_risky,
        "crime_types": {k: int(v) for k, v in crime_types.items()},
        "average_risk": round(avg_risk, 1),
        "monitored_locations": len(location_cache),
        "data_date_range": {
            "start": crime_data['datetime'].min().isoformat() if 'datetime' in crime_data.columns else None,
            "end": crime_data['datetime'].max().isoformat() if 'datetime' in crime_data.columns else None
        }
    }

# Serve static files (frontend)
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))