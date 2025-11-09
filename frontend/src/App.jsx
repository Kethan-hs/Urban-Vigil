import React, {useEffect, useState, useRef} from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const API_BASE = import.meta.env.VITE_API_BASE || "https://urban-vigil.onrender.com/api";

function TopNav(){ return (
  <nav className="w-full py-3 bg-gradient-to-r from-slate-900 to-slate-800 text-white shadow">
    <div className="max-w-6xl mx-auto px-4 flex items-center justify-between">
      <div><h1 className="text-xl font-bold">Urban Vigil Pro</h1><div className="text-sm text-slate-300">Bengaluru Safety Dashboard</div></div>
      <div className="flex gap-3 items-center">
        <a className="text-sm" href="#dashboard">Dashboard</a>
        <a className="text-sm" href="#map">Map</a>
        <a className="text-sm" href="#predict">Predict</a>
      </div>
    </div>
  </nav>
)}

function App(){
  const [mapReady, setMapReady] = useState(false);
  const mapRef = useRef(null);
  const markersRef = useRef(null);
  const [places, setPlaces] = useState([]);
  const [heatmap, setHeatmap] = useState([]);
  const [dashboard, setDashboard] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(()=>{ initMap(); loadPlaces(); loadDashboard(); loadHeatmap(); getDefaultPrediction(); },[]);

  async function loadPlaces(){
    try{
      const r = await fetch(API_BASE + "/place-lookup");
      const j = await r.json();
      setPlaces(j.places || []);
    }catch(e){ console.error(e); }
  }

  async function loadHeatmap(){
    try{
      const r = await fetch(API_BASE + "/heatmap");
      const j = await r.json();
      setHeatmap(j.heatmap || []);
      renderMarkers(j.heatmap || []);
    }catch(e){ console.error("Heatmap load error", e); }
  }

  async function loadDashboard(){
    try{
      const r = await fetch(API_BASE + "/dashboard");
      const j = await r.json();
      setDashboard(j);
    }catch(e){ console.error(e); }
  }

  async function getDefaultPrediction(){
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(async (pos)=>{
      const lat = pos.coords.latitude, lon = pos.coords.longitude;
      const r = await fetch(`${API_BASE}/predict?lat=${lat}&lon=${lon}`);
      const j = await r.json();
      setResult(j);
      if (mapRef.current) mapRef.current.setView([lat,lon],13);
    }, ()=>{});
  }

  function initMap(){
    if (mapRef.current) return;
    const map = L.map("mapid", {zoomControl:true}).setView([12.97,77.59],11);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {maxZoom:19}).addTo(map);
    mapRef.current = map;
    markersRef.current = L.layerGroup().addTo(map);
    setMapReady(true);
  }

  function renderMarkers(items){
    if(!mapRef.current) return;
    markersRef.current.clearLayers();
    items.forEach(h=>{
      const color = h.risk_score>=70? '#ef4444' : h.risk_score>=45? '#f59e0b':'#22c55e';
      const size = 8 + Math.min(26, Math.round(h.risk_score/4));
      const el = L.divIcon({className:'', html:`<div style="background:${color};width:${size}px;height:${size}px;border-radius:50%;border:2px solid rgba(255,255,255,0.9)"></div>`, iconSize:[size,size]});
      const m = L.marker([h.lat,h.lon], {icon:el}).addTo(markersRef.current);
      m.bindPopup(`<b>Risk: ${h.risk_score}</b><br/>Most common: ${h.most_common}<br/>Count: ${h.crime_count}`);
    });
    if (items.length) mapRef.current.fitBounds(items.map(i=>[i.lat,i.lon]), {padding:[60,60]});
  }

  async function onPredict(e){
    e.preventDefault();
    const form = new FormData(e.target);
    let place = form.get("place");
    let lat = form.get("lat"); let lon = form.get("lon"); let time = form.get("time");
    let q = "";
    if (place) q += `place=${encodeURIComponent(place)}`;
    if (lat && lon) q += `${q? '&':''}lat=${lat}&lon=${lon}`;
    if (time) q += `${q? '&':''}time=${encodeURIComponent(time)}`;
    const r = await fetch(API_BASE + "/predict?"+q);
    const j = await r.json();
    setResult(j);
    if (j.latitude && mapRef.current) mapRef.current.setView([j.latitude,j.longitude],14);
  }

  async function onRoute(e){
    e.preventDefault();
    const form = new FormData(e.target);
    const src = form.get("src"); const dst = form.get("dst");
    // src/dst can be place names; backend resolves
    const body = {src, dst};
    const r = await fetch(API_BASE + "/safe-route", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body)});
    const j = await r.json();
    // draw route polyline
    if (j.route_points){
      const coords = j.route_points.map(p=>[p.lat,p.lon]);
      L.polyline(coords, {color:'#7c3aed', weight:4}).addTo(markersRef.current);
      mapRef.current.fitBounds(coords, {padding:[60,60]});
    }
  }

  return (<div className="min-h-screen bg-slate-900 text-white">
    <TopNav/>
    <div className="max-w-6xl mx-auto p-4 grid md:grid-cols-3 gap-4">
      <div className="col-span-2 space-y-4">
        <section id="map" className="glass p-3 rounded">
          <div id="mapid" style={{height: "520px"}}></div>
        </section>

        <section id="predict" className="glass p-4 rounded mt-4">
          <h3 className="font-bold text-lg mb-2">Predict Safety</h3>
          <form onSubmit={onPredict} className="grid md:grid-cols-3 gap-2">
            <select name="place" className="p-2 bg-slate-800 rounded">
              <option value="">-- Choose place (or provide coords) --</option>
              {places.map(p=> <option key={p.Place} value={p.Place}>{p.Place}</option>)}
            </select>
            <input name="lat" placeholder="Latitude" className="p-2 bg-slate-800 rounded" />
            <input name="lon" placeholder="Longitude" className="p-2 bg-slate-800 rounded" />
            <input name="time" type="time" className="p-2 bg-slate-800 rounded md:col-span-1" />
            <div className="md:col-span-3 flex gap-2">
              <button className="px-4 py-2 bg-indigo-600 rounded">Predict</button>
              <button type="button" onClick={getDefaultPrediction} className="px-4 py-2 bg-gray-700 rounded">Use my location</button>
            </div>
          </form>
          {result && <div className="mt-3 p-3 bg-slate-800 rounded">
            <div><b>Predicted:</b> {result.predicted_crime} ({result.confidence}%)</div>
            <div><b>Risk score:</b> {result.risk_score}</div>
            <div><b>Nearest data:</b> {result.nearest? result.nearest.get('most_common','') : 'N/A'}</div>
          </div>}
        </section>

        <section id="route" className="glass p-4 rounded mt-4">
          <h3 className="font-bold text-lg mb-2">Safe Route Planner</h3>
          <form onSubmit={onRoute} className="grid md:grid-cols-3 gap-2">
            <select name="src" className="p-2 bg-slate-800 rounded">
              <option value="">-- Source place --</option>
              {places.map(p=> <option key={p.Place} value={p.Place}>{p.Place}</option>)}
            </select>
            <select name="dst" className="p-2 bg-slate-800 rounded">
              <option value="">-- Destination place --</option>
              {places.map(p=> <option key={p.Place+'d'} value={p.Place}>{p.Place}</option>)}
            </select>
            <div className="flex gap-2">
              <button className="px-4 py-2 bg-indigo-600 rounded">Get Route</button>
              <button type="button" className="px-4 py-2 bg-gray-700 rounded" onClick={()=>{ if (navigator.geolocation) navigator.geolocation.getCurrentPosition(p=>{ const lat=p.coords.latitude; const lon=p.coords.longitude; document.querySelector('select[name=src]').value=''; document.querySelector('input[name=src_lat]')?.setAttribute('value',lat); })}}>Use my location</button>
            </div>
          </form>
        </section>

      </div>

      <aside className="space-y-4">
        <section className="glass p-4 rounded">
          <h4 className="font-semibold">Dashboard</h4>
          {dashboard? <div className="mt-2 space-y-2">
            <div>Total crimes: {dashboard.total_crimes}</div>
            <div>Average risk: {dashboard.average_risk}</div>
            <div>Places monitored: {dashboard.places_monitored}</div>
            <div className="mt-2">
              <b>Top risky</b>
              <ul className="list-disc ml-5">
                {dashboard.top_risky.map((t,i)=> <li key={i}>{t.most_common || t.place} — {t.risk_score}</li>)}
              </ul>
            </div>
          </div> : <div>Loading...</div>}
        </section>

        <section className="glass p-4 rounded">
          <h4 className="font-semibold">Safety Tips</h4>
          <ul className="list-disc ml-5 mt-2 text-sm">
            <li>Share location with friends when travelling late.</li>
            <li>Prefer well-lit routes and populated areas.</li>
            <li>Keep emergency numbers handy.</li>
          </ul>
        </section>

      </aside>

    </div>
  </div>);
}

export default App;