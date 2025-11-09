Urban-Vigil Pro with OSRM Integration
=======================================
This version integrates the OSRM Route API for realistic road-based safe-route visualization.

Backend:
- Uses FastAPI
- New endpoint `/api/safe-route` calls `https://router.project-osrm.org` to compute routes
- If OSRM fails, falls back to straight-line route

Frontend:
- Draws OSRM route geometry using Leaflet's `L.geoJSON()`

Deploy same as previous version.
