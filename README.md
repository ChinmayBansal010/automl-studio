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

### Windows (CMD)
Backend terminal:
```bat
cd automl-studio\backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python run.py
```

Frontend in a second CMD:
```bat
cd automl-studio\frontend
npm install
npm run dev
```

### Linux (Konsole)
Backend terminal:
```bash
cd automl-studio/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

Frontend in a second Konsole:
```bash
cd automl-studio/frontend
npm install
npm run dev
```

Open http://localhost:5173. API docs: http://127.0.0.1:8000/docs.

## Local data
The backend saves datasets, models, and jobs under `backend/workspace/`. This directory is intentionally git-ignored; back it up separately if needed. The app is intended for local development and should not be exposed to untrusted networks without authentication.

## Notes
Training-time imputation, encoding, scaling, feature selection, and PCA are fitted inside the training pipeline to reduce leakage. Holdout metrics are estimates; repeatedly tuning against the same test split can overfit.