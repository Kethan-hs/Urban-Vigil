# Urban Vigil - Backend (Revamped)

This is a student project backend (FastAPI) for the Urban Vigil crime prediction app.

## Features implemented
- /predict : crime prediction (tries to use provided model, otherwise uses fallback heuristic)
- /heatmap : returns risk scores per place
- /risky-zones : top N risky places
- /dashboard : simple statistics
- /safe-route : improved safe-route suggestion (greedy waypoint insertion using haversine distances)
- /place-list : available places
- /health : service & model status

## Run locally
1. Create virtualenv and install:
```
pip install -r requirements.txt
```
2. Run:
```
uvicorn main:app --reload --host 0.0.0.0 --port 8080
```

Model file `crime_predictor_bangalore_new.pkl` (if present) will be loaded at startup. If model fails to load or is incompatible, the API will continue to work using a rule-based heuristic.
