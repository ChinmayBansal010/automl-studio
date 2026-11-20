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

## Run locally

### Configure authentication

The app uses Neon Managed Auth with email verification. `backend/.env.example` contains the two runtime settings the backend needs: the Neon Auth base URL (for the same-origin auth proxy) and its public JWKS URL (to verify sign-in tokens). Copy it to `backend/.env`; do not add a Neon management API key or database connection string. Neither is required by this app, and a management key must never be exposed in frontend code.

### Windows (CMD)
From the repository folder, open a CMD terminal for the backend:
```bat
cd automl-studio\backend
if not exist .venv python -m venv .venv
if not exist .env copy .env.example .env
.venv\Scripts\activate.bat
pip install -r requirements.txt
python run.py
```

Open a second CMD terminal for the frontend:
```bat
cd automl-studio\frontend
npm ci
npm run dev
```

Open http://localhost:5173 (or http://127.0.0.1:5173), create an account, and verify your email with the code Neon sends you. Both local origins are enabled for Neon Auth. The backend/API runs at http://127.0.0.1:8000 (API docs: http://127.0.0.1:8000/docs). Keep both terminals open while using the app.

### Linux (Konsole)
Backend terminal:
```bash
cd automl-studio/backend
test -d .venv || python3 -m venv .venv
test -f .env || cp .env.example .env
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Frontend in a second Konsole:
```bash
cd automl-studio/frontend
npm ci
npm run dev
```

Open http://localhost:5173 (or http://127.0.0.1:5173), create an account, and verify your email with the code Neon sends you. Both local origins are enabled for Neon Auth. The backend/API runs at http://127.0.0.1:8000 (API docs: http://127.0.0.1:8000/docs). Keep both terminals open while using the app.

## Deploy (free-tier configuration)

- Deploy `render.yaml` as a Render Blueprint. The API validates Neon Auth JWTs and keeps each signed-in user's workspace separate.
- Create a Vercel project with `frontend/` as its root directory. `frontend/vercel.json` sends `/api` and `/auth` through the Render API so auth cookies stay same-origin.
- After Vercel assigns the production URL, add that exact `https://...vercel.app` origin to Neon Auth's trusted domains. Keep `allow_localhost` enabled only if local development should continue using the production auth project.
- Render free services can sleep and have ephemeral disks. Uploaded datasets, trained models, jobs, and prediction exports on this tier can disappear after a restart or redeploy; use persistent/object storage before relying on long-term retention. Free-tier compute and email delivery are also limited.

The Render rewrite host in `frontend/vercel.json` assumes the API's Render service URL is `automl-studio-api.onrender.com`; update it if Render assigns a different URL. Account linking and the final trusted production domain are required before the app can be published safely.

## Local data
The backend stores datasets, models, jobs, and exports under `backend/workspace/users/`, in a separate directory per authenticated account. This directory is intentionally git-ignored; back it up separately if needed.

## Notes
Training-time imputation, encoding, scaling, feature selection, and PCA are fitted inside the training pipeline to reduce leakage. Holdout metrics are estimates; repeatedly tuning against the same test split can overfit.