# AutoML Studio

A local-first machine-learning workspace built with React, TypeScript, Vite, FastAPI, Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.

## Features
- CSV upload, schema inference, detailed dataset shape and column diagnostics.
- Missing-value handling (mean, median, mode, constant, zero, forward/backward fill), duplicate removal, row/column dropping, high-null filtering, IQR outlier removal/clipping, type conversion, renaming, sampling, and index reset.
- Feature engineering: one-hot encoding, standardization, min-max scaling, log transform, squared features, and PCA.
- Interactive EDA and a configurable chart builder for bar, line, area, scatter, pie, count, histogram, and correlation heatmap; configure axes, aggregation, row limit, and layout columns.
- Classification/regression AutoML and manual model selection; Random Forest, Extra Trees, Decision Tree, Gradient Boosting, Histogram Gradient Boosting, AdaBoost, KNN, SVM/SVR, Naive Bayes, Logistic/Ridge Regression, and MLP neural networks.
- SelectKBest, constant-feature filtering, PCA inside the training pipeline, holdout metrics, leaderboard, confusion matrix, model download, and external CSV predictions.
- Local persistence for datasets, trained pipelines, and job summaries in `backend/workspace/`; saved datasets can be restored after restarting the API.

---

## Prerequisites

| Tool | Version |
|------|---------|
| **Python** | 3.10+ |
| **Node.js** | 18+ (with npm) |
| **Git** | any recent version |

You also need a **Neon Auth** project (free tier works) for authentication.

---

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/automl-studio.git
cd automl-studio
```

### 2. Configure authentication

The app uses **Neon Managed Auth** with email verification. The backend needs two environment variables:

| Variable | Description |
|----------|-------------|
| `NEON_AUTH_BASE_URL` | Neon Auth base URL for the same-origin auth proxy |
| `NEON_AUTH_JWKS_URL` | Public JWKS endpoint to verify sign-in tokens |

Copy the example env files (do **not** commit secrets):

```bash
# Backend
cp backend/.env.example backend/.env

# Frontend
cp frontend/.env.example frontend/.env.local
```

> **Note**: Neither a Neon management API key nor a database connection string is required. Never expose a management key in frontend code.

### 3. Start the backend (Python / FastAPI)

#### Windows (CMD)
```bat
cd backend
if not exist .venv python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python run.py
```

#### Linux / macOS
```bash
cd backend
test -d .venv || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

The API starts at **http://127.0.0.1:8000** (Swagger docs: http://127.0.0.1:8000/docs).

### 4. Start the frontend (React / Vite)

Open a **second terminal**:

#### Windows (CMD)
```bat
cd frontend
npm ci
npm run dev
```

#### Linux / macOS
```bash
cd frontend
npm ci
npm run dev
```

The frontend starts at **http://localhost:5173** (or http://127.0.0.1:5173).

### 5. Use the app

1. Open http://localhost:5173 in your browser.
2. Create an account and verify your email with the OTP code Neon sends.
3. Upload a CSV and start exploring!

Both `localhost` and `127.0.0.1` origins are enabled for local Neon Auth. Keep both terminals open while using the app. Set `VITE_BACKEND_URL` in `frontend/.env.local` to override the backend URL if needed.

---

## Deploy to production

This section covers a free-tier deployment using **Render** (backend) + **Vercel** (frontend).

### Architecture

```
┌────────────────────────┐       HTTPS        ┌──────────────────────┐
│   Vercel (Frontend)    │  ◄──────────────►  │   Render (Backend)   │
│   React + Vite SSG     │                     │   FastAPI + uvicorn  │
│                        │  /auth/* rewrite    │                      │
│   vercel.json rewrites ├──────────────────►  │   /auth/* proxy ──► Neon Auth
└────────────────────────┘                     └──────────────────────┘
```

### Step 1: Set up Neon Auth

1. Sign up at [neon.tech](https://neon.tech) and create a project.
2. Enable **Managed Auth** in the Neon Console.
3. Note your **Auth Base URL** and **JWKS URL** from the auth settings.
4. Under trusted domains, add your future Vercel production URL (e.g. `https://your-project.vercel.app`). Keep `allow_localhost` enabled for local development.

### Step 2: Deploy the backend on Render

#### Option A: Render Blueprint (recommended)

1. Fork this repository to your GitHub account.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect your forked repo. Render reads `render.yaml` automatically.
4. Set these environment variables in the Render service settings:

| Variable | Value |
|----------|-------|
| `NEON_AUTH_BASE_URL` | Your Neon Auth base URL |
| `NEON_AUTH_JWKS_URL` | Your Neon Auth JWKS URL |
| `FRONTEND_ORIGINS` | Your Vercel URL, e.g. `https://your-project.vercel.app` (comma-separated; include `http://localhost:5173` if testing locally against Render) |

5. Deploy. Note the service URL (e.g. `https://automl-studio-api-xxxx.onrender.com`).

#### Option B: Manual Render setup

1. **New** → **Web Service** → connect your repo.
2. Settings:
   - **Root Directory**: `backend`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
   - **Health Check Path**: `/api/health`
3. Add the environment variables listed above.
4. Deploy.

### Step 3: Deploy the frontend on Vercel

1. Go to [vercel.com](https://vercel.com) → **Add New Project** → import your forked repo.
2. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Vite
   - **Build Command**: `npm run build` (default)
   - **Output Directory**: `dist` (default)
3. Add an environment variable:

| Variable | Value |
|----------|-------|
| `VITE_BACKEND_URL` | Your Render backend URL (e.g. `https://automl-studio-api-xxxx.onrender.com`) |

4. Deploy. Note your production URL.

### Step 4: Connect everything

1. **Update `frontend/.env.production`**: set `VITE_BACKEND_URL` to your Render URL.
2. **Update `frontend/vercel.json`**: update the rewrite destination to your Render URL:
   ```json
   {
     "rewrites": [
       {
         "source": "/auth/:path*",
         "destination": "https://YOUR-RENDER-URL.onrender.com/auth/:path*"
       }
     ]
   }
   ```
3. **Update Render's `FRONTEND_ORIGINS`**: add your Vercel URL so CORS works.
4. **Update Neon Auth trusted domains**: add your Vercel URL.
5. Redeploy both services.

### Step 5: Verify the deployment

1. Open your Vercel URL.
2. Create an account → verify email → sign in.
3. Upload a CSV and train a model.
4. Check the Render logs for any errors.

### Production checklist

- [ ] `FRONTEND_ORIGINS` on Render includes the exact Vercel origin
- [ ] `VITE_BACKEND_URL` on Vercel points to the Render API
- [ ] `vercel.json` rewrite destination matches the Render URL
- [ ] Neon Auth trusted domains include the Vercel origin
- [ ] Email verification is working (check Neon Auth email settings)

### Free-tier limitations

> **Important**: Render free services can sleep after 15 minutes of inactivity and have ephemeral disks. Uploaded datasets, trained models, jobs, and prediction exports on this tier **can disappear** after a restart or redeploy. Use persistent/object storage before relying on long-term retention. Free-tier compute and email delivery are also limited.

---

## Environment variables reference

### Backend (`backend/.env`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEON_AUTH_BASE_URL` | Yes | — | Neon Auth base URL |
| `NEON_AUTH_JWKS_URL` | No | `{NEON_AUTH_BASE_URL}/.well-known/jwks.json` | JWKS endpoint |
| `FRONTEND_ORIGINS` | No | `http://localhost:5173,http://127.0.0.1:5173` | Allowed CORS origins (comma-separated) |

### Frontend (`frontend/.env` / `.env.local` / `.env.production`)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_BACKEND_URL` | No | `http://127.0.0.1:8000` (dev) | Backend API base URL |

---

## Project structure

```
automl-studio/
├── backend/
│   ├── app/
│   │   └── main.py          # FastAPI application (all endpoints)
│   ├── workspace/            # Per-user data storage (git-ignored)
│   ├── tests/
│   │   └── test_auth.py      # Auth endpoint tests
│   ├── .env.example           # Backend env template
│   ├── requirements.txt       # Python dependencies
│   └── run.py                 # Dev server entry point
├── frontend/
│   ├── src/
│   │   ├── App.tsx            # Main app with workspace UI
│   │   ├── AuthScreen.tsx     # Login / signup / verification
│   │   ├── auth.ts            # Neon Auth client setup
│   │   ├── styles.css         # All styles
│   │   └── main.tsx           # React entry point
│   ├── .env.example           # Frontend env template
│   ├── .env.production        # Production backend URL
│   ├── vercel.json            # Vercel rewrites for auth proxy
│   ├── vite.config.ts         # Vite config with dev proxy
│   └── package.json           # Node dependencies
├── render.yaml                # Render Blueprint config
├── sample_data.csv            # Sample dataset
└── README.md                  # This file
```

## Local data

The backend stores datasets, models, jobs, and exports under `backend/workspace/users/`, in a separate directory per authenticated account. This directory is intentionally git-ignored; back it up separately if needed.

## Notes

Training-time imputation, encoding, scaling, feature selection, and PCA are fitted inside the training pipeline to reduce leakage. Holdout metrics are estimates; repeatedly tuning against the same test split can overfit.
