
# Urban Vigil Pro (Deploy-Ready Version)

## 🚀 Quick Deploy Instructions

### Backend (Render)
1. Sign up at [Render.com](https://render.com)
2. Create a **New Web Service** → Connect this GitHub repository
3. Render auto-detects `render.yaml`
4. Deploy → it builds & starts automatically
5. Note your backend URL (example): `https://urban-vigil-api.onrender.com`

### Frontend (Vercel)
1. Sign up at [Vercel.com](https://vercel.com)
2. Click **New Project → Import GitHub Repo**
3. Set **Root Directory** = `frontend`
4. Vercel auto-reads `vercel.json` and deploys instantly

✅ Your frontend will automatically connect to backend at:
`https://urban-vigil-api.onrender.com`

---

### Local Testing
```bash
# Run backend locally
uvicorn main:app --reload --port 8080

# Serve frontend locally
cd frontend
python -m http.server 8000
```
Then open: http://localhost:8000

---

Developed for final-year engineering project — Urban Vigil Pro (2025)
