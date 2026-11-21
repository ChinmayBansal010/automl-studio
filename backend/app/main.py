from __future__ import annotations

import gc, hashlib, io, json, os, re, threading, time, uuid, warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import joblib
import jwt
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from jwt.exceptions import PyJWKClientConnectionError
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore", category=UserWarning)
app = FastAPI(title="AutoML Studio API", version="2.0.0")
allowed_origins = [origin.strip() for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
RATE_LIMITS: dict[tuple[str, str], tuple[float, int]] = {}
RATE_LIMIT_LOCK = threading.Lock()

@app.middleware("http")
async def api_rate_limit(request: Request, call_next):
    path = request.url.path
    if not path.startswith("/api/"):
        return await call_next(request)
    authorization = request.headers.get("authorization", "")
    identity = hashlib.sha256(authorization.encode("utf-8")).hexdigest() if authorization else (request.client.host if request.client else "unknown")
    if path == "/api/train" and request.method == "POST":
        limit = 5
    elif path == "/api/datasets" and request.method == "POST":
        limit = 10
    elif path.endswith("/preprocess") or path.endswith("/feature-engineering"):
        limit = 30
    elif "/training-jobs/" in path and request.method == "GET":
        limit = 90
    else:
        limit = 120
    window = 60.0
    now = time.monotonic()
    route_key = path
    if path.startswith("/api/datasets/"):
        parts = path.strip("/").split("/")
        if len(parts) >= 4 and parts[3] in {"preprocess", "feature-engineering", "chart", "eda"}:
            route_key = "/api/datasets/{dataset_id}/" + parts[3]
    elif path.startswith("/api/training-jobs/"):
        route_key = "/api/training-jobs/{job_id}"
    key = (identity, route_key)
    with RATE_LIMIT_LOCK:
        started, count = RATE_LIMITS.get(key, (now, 0))
        if now - started >= window:
            started, count = now, 0
        if count >= limit:
            retry_after = max(1, int(window - (now - started)))
            return JSONResponse(status_code=429, content={"detail": "Too many requests. Please wait briefly and try again."}, headers={"Retry-After": str(retry_after)})
        RATE_LIMITS[key] = (started, count + 1)
        if len(RATE_LIMITS) > 10000:
            RATE_LIMITS.clear()
    return await call_next(request)

ROOT = Path(__file__).resolve().parent.parent / "workspace"
NEON_AUTH_BASE_URL = os.getenv("NEON_AUTH_BASE_URL", "").rstrip("/")
NEON_AUTH_JWKS_URL = os.getenv("NEON_AUTH_JWKS_URL", f"{NEON_AUTH_BASE_URL}/.well-known/jwks.json")
auth_url = urlsplit(NEON_AUTH_BASE_URL)
AUTH_ORIGIN = f"{auth_url.scheme}://{auth_url.netloc}" if auth_url.scheme == "https" and auth_url.netloc else ""
JWKS_CLIENT = jwt.PyJWKClient(NEON_AUTH_JWKS_URL, cache_jwk_set=True, lifespan=300) if AUTH_ORIGIN else None
datasets: dict[tuple[str, str], pd.DataFrame] = {}
models: dict[tuple[str, str], dict[str, Any]] = {}
LIMIT = 100 * 1024 * 1024

def user_workspace(user_id: str) -> tuple[Path, Path, Path, Path]:
    user_root = ROOT / "users" / hashlib.sha256(user_id.encode("utf-8")).hexdigest()
    paths = tuple(user_root / name for name in ("datasets", "models", "jobs", "exports"))
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths

def artifact_path(folder: Path, artifact_id: str, suffix: str) -> Path:
    try:
        if str(uuid.UUID(artifact_id)) != artifact_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(404, "Artifact not found.")
    return folder / f"{artifact_id}{suffix}"

def save_dataset_version(user_id: str, dataset_id: str, df: pd.DataFrame, filename: str, reason: str) -> dict[str, Any]:
    data_dir, _, _, _ = user_workspace(user_id)
    version_dir = data_dir / f"{dataset_id}.versions"
    version_dir.mkdir(parents=True, exist_ok=True)
    version_id = str(uuid.uuid4())
    csv_path = artifact_path(version_dir, version_id, ".csv")
    df.to_csv(csv_path, index=False)
    metadata = {
        "version_id": version_id,
        "dataset_id": dataset_id,
        "filename": filename,
        "reason": reason,
        "created_at": datetime.now().isoformat(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "size_bytes": csv_path.stat().st_size,
    }
    artifact_path(version_dir, version_id, ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    versions = sorted(version_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in versions[25:]:
        try:
            old_meta = json.loads(old.read_text(encoding="utf-8"))
            artifact_path(version_dir, old_meta["version_id"], ".csv").unlink(missing_ok=True)
            old.unlink(missing_ok=True)
        except Exception:
            continue
    return metadata

def dataset_filename(data_dir: Path, dataset_id: str) -> str:
    meta_path = artifact_path(data_dir, dataset_id, ".json")
    try:
        original = json.loads(meta_path.read_text(encoding="utf-8")).get("filename", "dataset.csv")
    except (OSError, ValueError):
        original = "dataset.csv"
    stem = Path(str(original)).stem
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("._-")[:80] or "dataset"
    return safe + "-processed.csv"

def current_user(authorization: str | None = Header(default=None)) -> str:
    if JWKS_CLIENT is None:
        raise HTTPException(503, "Authentication is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "A valid bearer token is required.")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(401, "A valid bearer token is required.")

    def _verify(allow_cache_refresh: bool = False):
        try:
            signing_key = JWKS_CLIENT.get_signing_key_from_jwt(token)
        except jwt.PyJWKClientError:
            if allow_cache_refresh:
                # Force a JWKS cache refresh and retry (key may have been rotated)
                JWKS_CLIENT.fetch_data()
                signing_key = JWKS_CLIENT.get_signing_key_from_jwt(token)
            else:
                raise
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["EdDSA"],
            issuer=AUTH_ORIGIN,
            audience=AUTH_ORIGIN,
            options={"require": ["sub", "iat", "exp"]},
        )

    try:
        claims = _verify(allow_cache_refresh=True)
    except PyJWKClientConnectionError as exc:
        raise HTTPException(503, "The authentication provider is temporarily unavailable.") from exc
    except (jwt.InvalidTokenError, jwt.PyJWKClientError, ValueError) as exc:
        raise HTTPException(401, "The bearer token is invalid or expired.") from exc
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        raise HTTPException(401, "The bearer token has no user identity.")
    if claims.get("emailVerified") is not True:
        raise HTTPException(403, "Verify your email address before using the workspace.")
    return user_id

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    names, seen = [], {}
    for i, col in enumerate(df.columns):
        name = str(col).strip() or f"column_{i+1}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    df.columns = names
    for c in df.columns:
        if df[c].dtype == object and any(k in c.lower() for k in ("date", "time", "timestamp")):
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() >= .7: df[c] = parsed
    return df

def get_df(dataset_id: str, user_id: str) -> pd.DataFrame:
    key = (user_id, dataset_id)
    data_dir, _, _, _ = user_workspace(user_id)
    if key not in datasets:
        path = artifact_path(data_dir, dataset_id, ".csv")
        if path.exists(): datasets[key] = clean(pd.read_csv(path, low_memory=False))
        else: raise HTTPException(404, "Dataset not found. Upload or restore the CSV.")
    return datasets[key]

def serial(v: Any) -> Any:
    if pd.isna(v): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (pd.Timestamp, datetime)): return v.isoformat()
    if isinstance(v, np.bool_): return bool(v)
    return v if isinstance(v, (str, int, float, bool)) else str(v)

def profile(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=np.number)
    columns = []
    for c in df.columns:
        s = df[c]
        kind = "datetime" if pd.api.types.is_datetime64_any_dtype(s) else "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical"
        columns.append({"column": c, "dtype": str(s.dtype), "kind": kind, "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean()*100), 2), "unique": int(s.nunique(dropna=True)),
            "sample": [serial(v) for v in s.dropna().head(3).tolist()]})
    stats = {}
    for c in numeric:
        s = numeric[c].dropna()
        q1, q3 = (float(s.quantile(.25)), float(s.quantile(.75))) if len(s) else (None, None)
        iqr = q3-q1 if q1 is not None and q3 is not None else None
        stats[c] = {"count":int(s.count()),"mean":serial(s.mean()),"std":serial(s.std()),"min":serial(s.min()),
            "q1":q1,"median":serial(s.median()),"q3":q3,"max":serial(s.max()),"variance":serial(s.var()),
            "skewness":serial(s.skew()),"kurtosis":serial(s.kurt()),"zeros":int((s==0).sum()),"negative":int((s<0).sum()),
            "outliers_iqr":int(((s<q1-1.5*iqr)|(s>q3+1.5*iqr)).sum()) if iqr is not None else 0}
    return {"rows":len(df),"columns":len(df.columns),"shape":[len(df),len(df.columns)],"total_cells":int(df.size),
        "duplicate_rows":int(df.duplicated().sum()),"duplicate_percentage":round(float(df.duplicated().mean()*100),2) if len(df) else 0,
        "memory_mb":round(float(df.memory_usage(deep=True).sum()/1048576),2),"columns_info":columns,
        "types":{"numeric":numeric.columns.tolist(),"categorical":[c for c in df.columns if c not in numeric.columns]},
        "column_stats":stats,"missing_total":int(df.isna().sum().sum()),
        "constant_columns":[c for c in df if df[c].nunique(dropna=False)<=1],
        "preview":[{k:serial(v) for k,v in row.items()} for row in df.head(10).to_dict("records")]}

def available_memory_mb() -> float | None:
    try:
        limit_path=Path("/sys/fs/cgroup/memory.max")
        current_path=Path("/sys/fs/cgroup/memory.current")
        if limit_path.exists() and current_path.exists():
            limit_text=limit_path.read_text().strip()
            if limit_text=="max": return None
            limit=int(limit_text); current=int(current_path.read_text().strip())
            return max(0.0,(limit-current)/1048576)
    except (OSError,ValueError):
        return None
    return None

def candidates(task: str) -> dict[str, Any]:
    if task == "classification":
        return {"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced"),
            "Decision Tree":DecisionTreeClassifier(random_state=42,class_weight="balanced"),
            "Random Forest":RandomForestClassifier(n_estimators=30,max_depth=10,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1,class_weight="balanced"),
            "Extra Trees":ExtraTreesClassifier(n_estimators=30,max_depth=12,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1,class_weight="balanced"),
            "Gradient Boosting":GradientBoostingClassifier(random_state=42),
            "Histogram Gradient Boosting":HistGradientBoostingClassifier(random_state=42),
            "AdaBoost":AdaBoostClassifier(random_state=42),"K-Nearest Neighbors":KNeighborsClassifier(),
            "Support Vector Machine":SVC(kernel="linear",class_weight="balanced",probability=False),"Naive Bayes":GaussianNB(),
            "Neural Network (MLP)":MLPClassifier(hidden_layer_sizes=(32,16),max_iter=150,early_stopping=True,random_state=42)}
    return {"Ridge Regression":Ridge(),"Decision Tree":DecisionTreeRegressor(random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=30,max_depth=12,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1),
        "Extra Trees":ExtraTreesRegressor(n_estimators=30,max_depth=12,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1),
        "Gradient Boosting":GradientBoostingRegressor(random_state=42),
        "Histogram Gradient Boosting":HistGradientBoostingRegressor(random_state=42),
        "AdaBoost":AdaBoostRegressor(random_state=42),"K-Nearest Neighbors":KNeighborsRegressor(),
        "Support Vector Regression":SVR(),"Neural Network (MLP)":MLPRegressor(hidden_layer_sizes=(32,16),max_iter=150,early_stopping=True,random_state=42)}

def make_pipeline(X: pd.DataFrame, estimator: Any, scale: bool, selection: str = "none", k: int = 10, pca: int = 0, task: str = "classification") -> Pipeline:
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in X.columns if c not in nums]
    parts = []
    if nums:
        steps = [("imputer",SimpleImputer(strategy="median"))]
        if scale: steps.append(("scale",StandardScaler()))
        parts.append(("num",Pipeline(steps),nums))
    if cats: parts.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=5,max_categories=10,dtype=np.float32))]),cats))
    steps: list[tuple[str,Any]] = [("prep",ColumnTransformer(parts,verbose_feature_names_out=False))]
    if selection == "kbest": steps.append(("select",SelectKBest(f_classif if task=="classification" else f_regression,k=max(1,min(k,len(X.columns))))))
    if selection == "variance": steps.append(("select",VarianceThreshold()))
    if pca: steps.append(("pca",PCA(n_components=max(1,min(pca,len(X.columns),max(1,len(X)-1))),random_state=42)))
    steps.append(("model",estimator))
    return Pipeline(steps)

@app.get("/api/health")
def health(): return {"status":"ok","service":"AutoML Studio API"}

@app.api_route("/auth/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def auth_proxy(path: str, request: Request):
    if not NEON_AUTH_BASE_URL:
        raise HTTPException(503, "Neon Auth is not configured.")
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() in {"content-type", "cookie", "origin", "user-agent"}
    }
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
            for attempt in range(3):
                try:
                    upstream = await client.request(
                        request.method,
                        f"{NEON_AUTH_BASE_URL}/{path}",
                        params=request.query_params,
                        content=await request.body(),
                        headers=headers,
                    )
                    break
                except httpx.RequestError:
                    if attempt >= 2:
                        raise
                    import asyncio
                    await asyncio.sleep(1 * (attempt + 1))
    except httpx.RequestError as exc:
        raise HTTPException(502, "Neon Auth could not be reached. Please try again.") from exc

    response = Response(content=upstream.content, status_code=upstream.status_code)
    for name in ("content-type", "cache-control", "location", "set-auth-jwt"):
        if name in upstream.headers:
            response.headers[name] = upstream.headers[name]
    for cookie in upstream.headers.get_list("set-cookie"):
        cookie = re.sub(r";\s*Domain=[^;]+", "", cookie, flags=re.IGNORECASE)
        cookie = re.sub(r";\s*Path=[^;]+", "; Path=/auth", cookie, flags=re.IGNORECASE)
        response.headers.append("set-cookie", cookie)
    return response

@app.post("/api/datasets")
async def upload(file: UploadFile = File(...), user_id: str = Depends(current_user)):
    if not file.filename or not file.filename.lower().endswith(".csv"): raise HTTPException(400,"Upload a .csv file.")
    content = await file.read()
    if not content or len(content)>LIMIT: raise HTTPException(413 if content else 400,"CSV is empty or exceeds 100 MB.")
    try: df = clean(pd.read_csv(io.BytesIO(content),low_memory=False))
    except Exception as e: raise HTTPException(400,f"Could not parse CSV: {e}")
    if df.empty: raise HTTPException(400,"CSV has headers but no rows.")
    data_dir, _, _, _ = user_workspace(user_id)
    did = str(uuid.uuid4()); datasets[(user_id, did)] = df; df.to_csv(data_dir/f"{did}.csv",index=False)
    (data_dir/f"{did}.json").write_text(json.dumps({"dataset_id":did,"filename":file.filename,"saved_at":datetime.now().isoformat()}),encoding="utf-8")
    save_dataset_version(user_id, did, df, file.filename, "Uploaded original dataset")
    return {"dataset_id":did,"filename":file.filename,"saved_locally":True,**profile(df)}

@app.post("/api/datasets/{dataset_id}/restore")
async def restore_dataset_file(dataset_id: str, file: UploadFile = File(...), user_id: str = Depends(current_user)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400,"Upload a .csv file.")
    content=await file.read()
    if not content or len(content)>LIMIT:
        raise HTTPException(413 if content else 400,"CSV is empty or exceeds 100 MB.")
    try:
        df=clean(pd.read_csv(io.BytesIO(content),low_memory=False))
    except Exception as exc:
        raise HTTPException(400,"Could not restore the saved CSV: "+str(exc)) from exc
    if df.empty:
        raise HTTPException(400,"CSV has headers but no rows.")
    data_dir,_,_,_=user_workspace(user_id)
    path=artifact_path(data_dir,dataset_id,".csv")
    datasets[(user_id,dataset_id)]=df
    df.to_csv(path,index=False)
    metadata={"dataset_id":dataset_id,"filename":file.filename,"saved_at":datetime.now().isoformat()}
    artifact_path(data_dir,dataset_id,".json").write_text(json.dumps(metadata),encoding="utf-8")
    save_dataset_version(user_id, dataset_id, df, file.filename, "Restored browser backup")
    return {"dataset_id":dataset_id,"filename":file.filename,"saved_locally":True,**profile(df)}

@app.get("/api/datasets/{dataset_id}/download")
def download_dataset_file(dataset_id: str, user_id: str = Depends(current_user)):
    data_dir,_,_,_=user_workspace(user_id)
    path=artifact_path(data_dir,dataset_id,".csv")
    if not path.exists():
        raise HTTPException(404,"Dataset file is not available on this server. Restore it from the browser backup.")
    filename = dataset_filename(data_dir, dataset_id)
    return StreamingResponse(open(path,"rb"),media_type="text/csv; charset=utf-8",headers={"Content-Disposition":f'attachment; filename="{filename}"'})

@app.get("/api/datasets/{dataset_id}/versions")
def list_dataset_versions(dataset_id: str, user_id: str = Depends(current_user)):
    artifact_path(user_workspace(user_id)[0], dataset_id, ".csv")
    version_dir = user_workspace(user_id)[0] / f"{dataset_id}.versions"
    versions = []
    if version_dir.exists():
        for path in version_dir.glob("*.json"):
            try:
                versions.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return {"versions": sorted(versions, key=lambda item: item.get("created_at", ""), reverse=True)}

@app.post("/api/datasets/{dataset_id}/versions/{version_id}/restore")
def restore_dataset_version(dataset_id: str, version_id: str, user_id: str = Depends(current_user)):
    data_dir, _, _, _ = user_workspace(user_id)
    version_dir = data_dir / f"{dataset_id}.versions"
    metadata_path = artifact_path(version_dir, version_id, ".json")
    csv_path = artifact_path(version_dir, version_id, ".csv")
    if not metadata_path.is_file() or not csv_path.is_file():
        raise HTTPException(404, "Dataset version not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    df = clean(pd.read_csv(csv_path, low_memory=False))
    if df.empty:
        raise HTTPException(400, "This dataset version contains no rows.")
    datasets[(user_id, dataset_id)] = df
    df.to_csv(artifact_path(data_dir, dataset_id, ".csv"), index=False)
    save_dataset_version(user_id, dataset_id, df, metadata.get("filename", "dataset.csv"), "Restored version " + version_id[:8])
    return {"dataset_id": dataset_id, "filename": metadata.get("filename", "dataset.csv"), "saved_locally": True, **profile(df)}

@app.get("/api/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str, user_id: str = Depends(current_user)): return profile(get_df(dataset_id, user_id))

@app.get("/api/datasets/{dataset_id}/eda")
def eda(dataset_id: str, user_id: str = Depends(current_user)):
    df=get_df(dataset_id, user_id); numeric=df.select_dtypes(include=np.number); hist=[]
    for c in numeric.columns[:20]:
        vals=numeric[c].dropna()
        if len(vals):
            counts, edges=np.histogram(vals,bins=min(30,max(5,int(np.sqrt(len(vals))))))
            hist.append({"column":c,"bins":[{"range":f"{edges[i]:.4g}–{edges[i+1]:.4g}","count":int(counts[i])} for i in range(len(counts))]})
    cats=[]
    for c in [x for x in df.columns if x not in numeric.columns][:15]:
        counts=df[c].astype("string").fillna("(Missing)").value_counts().head(20)
        cats.append({"column":c,"items":[{"label":str(k),"count":int(v)} for k,v in counts.items()]})
    corr=numeric.corr().replace({np.nan:None}).round(3)
    return {"histograms":hist,"categorical":cats,"correlation":[{"column":str(i),**{str(k):serial(v) for k,v in row.items()}} for i,row in corr.iterrows()]}

class Prep(BaseModel):
    operation: str
    columns: list[str] = []
    strategy: str = "median"
    value: str = ""
    threshold: float = Field(default=.5,ge=0,le=1)
    new_name: str = ""
    components: int = Field(default=2,ge=1,le=100)
    sample_size: int = Field(default=1000,ge=1,le=500000)

@app.post("/api/datasets/{dataset_id}/preprocess")
def preprocess(dataset_id: str, req: Prep, user_id: str = Depends(current_user)):
    df = get_df(dataset_id, user_id).copy()
    cols = [c for c in req.columns if c in df]
    op = req.operation
    if op == "drop_duplicates":
        df = df.drop_duplicates()
    elif op == "drop_missing_rows":
        df = df.dropna(subset=cols or None)
    elif op == "drop_columns":
        if not cols: raise HTTPException(400, "Select at least one column to drop.")
        df = df.drop(columns=cols)
    elif op == "fill_missing":
        selected = cols or df.columns.tolist()
        if req.strategy not in {"mean","median","mode","constant","zero","ffill","bfill","drop"}:
            raise HTTPException(400, "Choose a valid missing-value strategy.")
        if req.strategy == "drop":
            df = df.dropna(subset=selected)
        elif req.strategy in {"mean","median","zero"}:
            numeric = [c for c in selected if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric: raise HTTPException(400, "This strategy requires at least one selected numeric column.")
            for c in numeric:
                fill_value = 0 if req.strategy == "zero" else df[c].mean() if req.strategy == "mean" else df[c].median()
                df[c] = df[c].fillna(fill_value)
        elif req.strategy == "mode":
            for c in selected:
                modes = df[c].mode(dropna=True)
                if not modes.empty: df[c] = df[c].fillna(modes.iloc[0])
        elif req.strategy == "constant":
            for c in selected: df[c] = df[c].fillna(req.value if req.value != "" else "Unknown")
        elif req.strategy == "ffill":
            df[selected] = df[selected].ffill()
        elif req.strategy == "bfill":
            df[selected] = df[selected].bfill()
    elif op == "drop_high_missing":
        removed = [c for c in df.columns if df[c].isna().mean() > req.threshold]
        if len(removed) == len(df.columns): raise HTTPException(400, "The selected threshold would remove every column.")
        df = df.drop(columns=removed)
    elif op in ("drop_outliers","clip_outliers"):
        selected = cols or df.select_dtypes(include=np.number).columns.tolist()
        numeric = [c for c in selected if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric: raise HTTPException(400, "Select at least one numeric column for IQR outlier handling.")
        for c in numeric:
            values = df[c].dropna()
            if values.empty: continue
            q1,q3 = values.quantile(.25),values.quantile(.75)
            lower,upper = q1-1.5*(q3-q1),q3+1.5*(q3-q1)
            if op == "clip_outliers": df[c] = df[c].clip(lower,upper)
            else: df = df[df[c].isna() | df[c].between(lower,upper)]
    elif op == "convert_numeric":
        if not cols: raise HTTPException(400, "Select at least one column to convert.")
        for c in cols: df[c] = pd.to_numeric(df[c],errors="coerce")
    elif op == "convert_datetime":
        if not cols: raise HTTPException(400, "Select at least one column to convert.")
        for c in cols: df[c] = pd.to_datetime(df[c],errors="coerce")
    elif op == "rename_column":
        if len(cols)!=1 or not req.new_name.strip(): raise HTTPException(400, "Select exactly one column and enter its new name.")
        new_name=req.new_name.strip()
        if new_name in df.columns and new_name!=cols[0]: raise HTTPException(400, "That column name already exists.")
        df=df.rename(columns={cols[0]:new_name})
    elif op == "sample_rows":
        df=df.sample(n=min(req.sample_size,len(df)),random_state=42)
    elif op == "reset_index":
        df=df.reset_index(drop=True)
    else:
        raise HTTPException(400, "Unknown preprocessing operation.")
    if df.empty: raise HTTPException(400, "This operation would create an empty dataset. No changes were saved.")
    data_dir,_,_,_=user_workspace(user_id)
    datasets[(user_id,dataset_id)]=df
    df.to_csv(artifact_path(data_dir,dataset_id,".csv"),index=False)
    save_dataset_version(user_id, dataset_id, df, dataset_filename(data_dir, dataset_id).removesuffix("-processed.csv") + ".csv", "Preprocessing: " + op)
    return {"dataset_id":dataset_id,"operation":op,"saved_locally":True,**profile(df)}

@app.post("/api/datasets/{dataset_id}/feature-engineering")
def feature_engineering(dataset_id: str, payload: dict[str,Any], user_id: str = Depends(current_user)):
    df=get_df(dataset_id,user_id).copy()
    op=payload.get("operation")
    cols=[c for c in payload.get("columns",[]) if c in df]
    if op=="pca":
        cols=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols)<2: raise HTTPException(400,"PCA requires at least two selected numeric columns.")
        x=df[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        x=x.fillna(x.median()).fillna(0)
        n=min(int(payload.get("components",2)),len(cols),len(df)-1)
        if n<1: raise HTTPException(400,"Not enough rows to calculate PCA.")
        z=PCA(n_components=n,random_state=42).fit_transform(StandardScaler().fit_transform(x))
        df=df.drop(columns=cols)
        for i in range(n): df["PC"+str(i+1)]=z[:,i]
    else:
        if not cols: raise HTTPException(400,"Select at least one column for feature engineering.")
        if op=="one_hot_encode":
            df=pd.get_dummies(df,columns=cols,dummy_na=bool(payload.get("dummy_na",False)),dtype=np.uint8)
        elif op in ("standardize","minmax","log_transform","polynomial"):
            numeric=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
            if not numeric: raise HTTPException(400,"This transformation requires at least one selected numeric column.")
            for c in numeric:
                values=pd.to_numeric(df[c],errors="coerce")
                if op=="standardize":
                    std=values.std()
                    df[c]=(values-values.mean())/(std if pd.notna(std) and std!=0 else 1)
                elif op=="minmax":
                    span=values.max()-values.min()
                    df[c]=(values-values.min())/(span if pd.notna(span) and span!=0 else 1)
                elif op=="log_transform": df[c]=np.log1p(values.clip(lower=0))
                else: df[c+"__squared"]=values**2
        else: raise HTTPException(400,"Unknown feature engineering operation.")
    data_dir,_,_,_=user_workspace(user_id)
    datasets[(user_id,dataset_id)]=df
    df.to_csv(artifact_path(data_dir,dataset_id,".csv"),index=False)
    save_dataset_version(user_id, dataset_id, df, dataset_filename(data_dir, dataset_id).removesuffix("-processed.csv") + ".csv", "Feature engineering: " + str(op))
    return {"dataset_id":dataset_id,"operation":op,"saved_locally":True,**profile(df)}

@app.post("/api/datasets/{dataset_id}/chart")
def chart(dataset_id: str, payload: dict[str,Any], user_id: str = Depends(current_user)):
    df=get_df(dataset_id, user_id)
    kind=payload.get("chart_type","bar")
    x=payload.get("x")
    y=payload.get("y")
    rows=max(1,min(int(payload.get("rows",20)),500))
    agg=payload.get("aggregation","mean")
    columns=max(1,min(int(payload.get("columns",1)),6))
    if kind not in {"bar","line","area","scatter","pie","count","histogram","heatmap","density","box","radar"}:
        raise HTTPException(400,"Choose a supported chart type.")
    if kind=="heatmap":
        numeric=df.select_dtypes(include=np.number).iloc[:,:12]
        if numeric.shape[1] < 2:
            return {"chart_type":kind,"x":None,"y":None,"x_label":None,"y_label":None,"rows":rows,"columns":columns,"data":[],"message":"Heatmap requires at least two numeric columns."}
        corr=numeric.corr().round(3).replace({np.nan:None})
        data=[{"x":str(i),**{str(k):serial(v) for k,v in r.items()}} for i,r in corr.iterrows()]
        return {"chart_type":kind,"x":None,"y":None,"x_label":None,"y_label":None,"rows":rows,"columns":columns,"data":data}
    if kind=="density":
        if x not in df or not pd.api.types.is_numeric_dtype(df[x]):
            raise HTTPException(400,"Choose a numeric column for the density chart.")
        vals=pd.to_numeric(df[x],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        counts,edges=np.histogram(vals,bins=min(rows,60),density=True) if len(vals) else ([],[])
        data=[{"x":float((edges[i]+edges[i+1])/2),"value":float(counts[i])} for i in range(len(counts))]
        return {"chart_type":kind,"x":x,"y":None,"x_label":x,"y_label":"Density","rows":rows,"columns":columns,"data":data}
    if kind=="box":
        if x not in df or y not in df or not pd.api.types.is_numeric_dtype(df[y]):
            raise HTTPException(400,"Box plots require a category X axis and numeric Y axis.")
        grouped=df.groupby(x,dropna=False)[y]
        data=[]
        for label,values in grouped:
            values=pd.to_numeric(values,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
            if values.empty: continue
            q1=float(values.quantile(.25)); median=float(values.median()); q3=float(values.quantile(.75))
            data.append({"x":str(label),"min":float(values.min()),"q1":q1,"median":median,"q3":q3,"max":float(values.max()),"range":[median-q1,q3-median]})
            if len(data)>=rows: break
        return {"chart_type":kind,"x":x,"y":y,"x_label":x,"y_label":y,"rows":rows,"columns":columns,"data":data}
    if kind=="radar":
        if x not in df or y not in df or not pd.api.types.is_numeric_dtype(df[y]):
            raise HTTPException(400,"Radar charts require a category X axis and numeric Y axis.")
        grouped=df.groupby(x,dropna=False)[y]
        values={"sum":grouped.sum(),"count":grouped.count(),"median":grouped.median(),"min":grouped.min(),"max":grouped.max()}.get(agg,grouped.mean()).head(rows)
        data=[{"x":str(label),"value":serial(value)} for label,value in values.items()]
        return {"chart_type":kind,"x":x,"y":y,"x_label":x,"y_label":y,"rows":rows,"columns":columns,"data":data}
    if kind=="histogram":
        column=x if x in df else y
        if column not in df or not pd.api.types.is_numeric_dtype(df[column]):
            raise HTTPException(400,"Choose a numeric column for the histogram.")
        vals=pd.to_numeric(df[column],errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
        counts,edges=np.histogram(vals,bins=min(rows,60)) if len(vals) else ([],[])
        data=[{"x":f"{edges[i]:.4g}–{edges[i+1]:.4g}","value":int(counts[i])} for i in range(len(counts))]
        return {"chart_type":kind,"x":column,"y":None,"x_label":column,"y_label":"Frequency","rows":rows,"columns":columns,"data":data}
    if x not in df:
        raise HTTPException(400,"Choose a valid category or X-axis column.")
    if kind in ("pie","count"):
        data=[{"x":str(k),"value":int(v)} for k,v in df[x].astype("string").fillna("(Missing)").value_counts().head(rows).items()]
        return {"chart_type":kind,"x":x,"y":None,"x_label":None,"y_label":None,"rows":rows,"columns":columns,"data":data}
    if kind=="scatter":
        if y not in df:
            raise HTTPException(400,"Choose a valid Y-axis column.")
        if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
            raise HTTPException(400,"Scatter plots require numeric X and Y columns.")
        pairs=pd.DataFrame({"x":pd.to_numeric(df[x],errors="coerce"),"value":pd.to_numeric(df[y],errors="coerce")}).replace([np.inf,-np.inf],np.nan).dropna().head(rows)
        data=[{"x":serial(a),"value":serial(b)} for a,b in zip(pairs["x"],pairs["value"])]
        return {"chart_type":kind,"x":x,"y":y,"x_label":x,"y_label":y,"rows":rows,"columns":columns,"data":data}
    if kind in ("line","area"):
        if y not in df:
            raise HTTPException(400,"Choose a valid Y-axis column.")
        if not pd.api.types.is_numeric_dtype(df[y]):
            raise HTTPException(400,"Choose a numeric Y-axis column.")
        grouped=df.groupby(x,dropna=False)[y]
        values={"sum":grouped.sum(),"count":grouped.count(),"median":grouped.median(),"min":grouped.min(),"max":grouped.max()}.get(agg,grouped.mean()).head(rows)
        data=[{"x":str(k),"value":serial(v)} for k,v in values.items()]
        return {"chart_type":kind,"x":x,"y":y,"x_label":x,"y_label":y,"rows":rows,"columns":columns,"data":data}
    if y in df and pd.api.types.is_numeric_dtype(df[y]):
        grouped=df.groupby(x,dropna=False)[y]
        values={"sum":grouped.sum(),"count":grouped.count(),"median":grouped.median(),"min":grouped.min(),"max":grouped.max()}.get(agg,grouped.mean()).head(rows)
        data=[{"x":str(k),"value":serial(v)} for k,v in values.items()]
        y_label=y
    else:
        data=[{"x":str(k),"value":int(v)} for k,v in df[x].astype("string").fillna("(Missing)").value_counts().head(rows).items()]
        y_label="Count"
    return {"chart_type":kind,"x":x,"y":y if y in df else None,"x_label":x,"y_label":y_label,"rows":rows,"columns":columns,"data":data}

class Train(BaseModel):
    dataset_id: str
    target: str
    task: str = Field(pattern="^(classification|regression)$")
    mode: str = Field(default="auto",pattern="^(auto|manual)$")
    model_name: str = "Random Forest"
    model_names: list[str] | None = None
    test_size: float = Field(default=.2,ge=.1,le=.5)
    random_state: int = 42
    scale_numeric: bool = True
    drop_high_missing: bool = False
    selected_features: list[str] | None = None
    feature_selection: str = Field(default="none",pattern="^(none|kbest|variance)$")
    k_features: int = Field(default=10,ge=1,le=500)
    use_pca: bool = False
    pca_components: int = Field(default=0,ge=0,le=500)

TRAINING_JOBS: dict[tuple[str, str], dict[str, Any]] = {}
TRAINING_CANCEL_EVENTS: dict[tuple[str, str], threading.Event] = {}
TRAINING_JOBS_LOCK = threading.Lock()
TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="automl-training")
MAX_QUEUED_JOBS_PER_USER = 3

def persist_training_record(user_id: str, record: dict[str, Any]) -> None:
    try:
        _, _, job_dir, _ = user_workspace(user_id)
        job_id = record.get("job_id")
        if job_id:
            durable_record = {key: value for key, value in record.items() if key != "result"}
            artifact_path(job_dir, job_id, ".json").write_text(json.dumps(durable_record, indent=2, default=str), encoding="utf-8")
    except (OSError, HTTPException):
        pass

def update_training_record(user_id: str, job_id: str, updates: dict[str, Any]) -> None:
    key = (user_id, job_id)
    with TRAINING_JOBS_LOCK:
        record = TRAINING_JOBS.get(key)
        if record is None:
            return
        record.update(updates)
        snapshot = dict(record)
    persist_training_record(user_id, snapshot)

def execute_training_job(req: Train, user_id: str, job_id: str):
    key = (user_id, job_id)
    cancel_event = TRAINING_CANCEL_EVENTS.get(key, threading.Event())
    with TRAINING_JOBS_LOCK:
        if key not in TRAINING_JOBS:
            return
        if cancel_event.is_set():
            TRAINING_JOBS[key].update({"status": "cancelled", "finished_at": datetime.now().isoformat()})
            return
        TRAINING_JOBS[key].update({"status": "running", "started_at": datetime.now().isoformat()})
        snapshot = dict(TRAINING_JOBS[key])
    persist_training_record(user_id, snapshot)
    try:
        result = run_training(req, user_id, job_id, cancel_event)
        with TRAINING_JOBS_LOCK:
            if key in TRAINING_JOBS:
                TRAINING_JOBS[key].update({
                    "status": "completed", "result": result, "error": None,
                    "finished_at": datetime.now().isoformat(), "progress_pct": 100,
                    "eta_seconds": 0, "model_id": result.get("model_id"),
                    "best_model": result.get("best_model"),
                    "train_rows": result.get("train_rows"), "test_rows": result.get("test_rows"),
                    "leaderboard": result.get("leaderboard"),
                    "best_evaluation": result.get("best_evaluation")
                })
                snapshot = dict(TRAINING_JOBS[key])
        persist_training_record(user_id, snapshot)
    except HTTPException as exc:
        with TRAINING_JOBS_LOCK:
            if key in TRAINING_JOBS:
                status = "cancelled" if cancel_event.is_set() else "failed"
                TRAINING_JOBS[key].update({"status": status, "error": None if status == "cancelled" else str(exc.detail), "finished_at": datetime.now().isoformat()})
                snapshot = dict(TRAINING_JOBS[key])
        persist_training_record(user_id, snapshot)
    except Exception as exc:
        with TRAINING_JOBS_LOCK:
            if key in TRAINING_JOBS:
                status = "cancelled" if cancel_event.is_set() else "failed"
                TRAINING_JOBS[key].update({"status": status, "error": None if status == "cancelled" else "Training failed: " + str(exc), "finished_at": datetime.now().isoformat()})
                snapshot = dict(TRAINING_JOBS[key])
        persist_training_record(user_id, snapshot)
    finally:
        gc.collect()

@app.post("/api/train")
def train(req: Train, user_id: str = Depends(current_user)):
    with TRAINING_JOBS_LOCK:
        active = [
            job for (owner, _), job in TRAINING_JOBS.items()
            if owner == user_id and job["status"] in {"queued", "running"}
        ]
        if len(active) >= MAX_QUEUED_JOBS_PER_USER:
            raise HTTPException(429, "You already have three queued or running jobs. Wait for one to finish before adding another.")
        job_id = str(uuid.uuid4())
        queue_position = sum(1 for job in TRAINING_JOBS.values() if job["status"] in {"queued", "running"}) + 1
        TRAINING_JOBS[(user_id, job_id)] = {
            "job_id": job_id, "status": "queued", "task": req.task, "target": req.target,
            "mode": req.mode, "model_name": req.model_name, "model_names": req.model_names, "queue_position": queue_position,
            "created_at": datetime.now().isoformat(), "error": None, "result": None,
            "request_config": req.model_dump(), "progress_pct": 0, "completed_models": 0,
            "total_models": len(req.model_names or []) if req.mode == "manual" else len(candidates(req.task)),
            "current_model": None, "eta_seconds": None
        }
        TRAINING_CANCEL_EVENTS[(user_id, job_id)] = threading.Event()
        queued_record = dict(TRAINING_JOBS[(user_id, job_id)])
    persist_training_record(user_id, queued_record)
    try:
        TRAINING_EXECUTOR.submit(execute_training_job, req, user_id, job_id)
    except Exception as exc:
        with TRAINING_JOBS_LOCK:
            TRAINING_JOBS[(user_id, job_id)].update({"status": "failed", "error": "Could not queue training: " + str(exc)})
        raise HTTPException(503, "Training queue is temporarily unavailable.") from exc
    return {"job_id": job_id, "status": "queued", "queue_position": queue_position, "message": "Training job added to the queue. Jobs run one at a time to protect the free server memory limit."}

@app.post("/api/training-jobs/{job_id}/retry")
def retry_training_job(job_id: str, user_id: str = Depends(current_user)):
    _, _, job_dir, _ = user_workspace(user_id)
    source_path = artifact_path(job_dir, job_id, ".json")
    if not source_path.is_file():
        raise HTTPException(404, "Saved experiment configuration was not found.")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    config = source.get("request_config")
    if not config:
        raise HTTPException(409, "This experiment does not contain a saved configuration and cannot be retried.")
    req = Train(**config)
    with TRAINING_JOBS_LOCK:
        active = [job for (owner, _), job in TRAINING_JOBS.items() if owner == user_id and job.get("status") in {"queued", "running"}]
        if len(active) >= MAX_QUEUED_JOBS_PER_USER:
            raise HTTPException(429, "You already have three queued or running jobs.")
        new_id = str(uuid.uuid4())
        record = {
            "job_id": new_id, "status": "queued", "task": req.task, "target": req.target,
            "mode": req.mode, "model_name": req.model_name, "model_names": req.model_names,
            "queue_position": len(active) + 1, "created_at": datetime.now().isoformat(),
            "error": None, "result": None, "request_config": req.model_dump(),
            "retry_of": job_id, "progress_pct": 0, "completed_models": 0,
            "total_models": len(req.model_names or []) if req.mode == "manual" else len(candidates(req.task)),
            "current_model": None, "eta_seconds": None
        }
        TRAINING_JOBS[(user_id, new_id)] = record
        TRAINING_CANCEL_EVENTS[(user_id, new_id)] = threading.Event()
    persist_training_record(user_id, record)
    TRAINING_EXECUTOR.submit(execute_training_job, req, user_id, new_id)
    return {"job_id": new_id, "status": "queued", "queue_position": record["queue_position"], "retry_of": job_id}

@app.get("/api/training-jobs/{job_id}")
def training_job_status(job_id: str, user_id: str = Depends(current_user)):
    try:
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(404, "Training job not found.")
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get((user_id, job_id))
        if job is None:
            raise HTTPException(404, "Training job not found or no longer available in this server session.")
        response = {key: value for key, value in job.items() if key != "result"}
        if job["status"] == "completed":
            response["result"] = job.get("result")
        return response

def cancel_training_job(job_id: str, user_id: str = Depends(current_user)):
    try:
        if str(uuid.UUID(job_id)) != job_id:
            raise ValueError
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(404, "Training job not found.")
    key = (user_id, job_id)
    with TRAINING_JOBS_LOCK:
        job = TRAINING_JOBS.get(key)
        if job is None:
            raise HTTPException(404, "Training job not found.")
        if job["status"] not in {"queued", "running"}:
            return {"job_id": job_id, "status": job["status"], "message": "This training job is already finished."}
        event = TRAINING_CANCEL_EVENTS.get(key)
        if event:
            event.set()
        if job["status"] == "queued":
            job.update({"status": "cancelled", "finished_at": datetime.now().isoformat()})
            message = "Queued training job cancelled."
        else:
            job["cancel_requested"] = True
            message = "Stop requested. The current model fit will finish safely, then remaining models will be skipped."
    return {"job_id": job_id, "status": "cancelled" if job["status"] == "cancelled" else "cancelling", "message": message}

@app.post("/api/training-jobs/{job_id}/cancel")
def cancel_training_job_route(job_id: str, user_id: str = Depends(current_user)):
    return cancel_training_job(job_id, user_id)

def run_training(req: Train, user_id: str, assigned_model_id: str | None = None, cancel_event: threading.Event | None = None):
    if cancel_event and cancel_event.is_set():
        raise HTTPException(409, "Training was cancelled.")
    source_df = get_df(req.dataset_id, user_id)
    if req.target not in source_df.columns:
        raise HTTPException(400, "Target column not found.")
    target_values = source_df[req.target]
    if req.task == "regression":
        if not pd.api.types.is_numeric_dtype(target_values):
            raise HTTPException(400, "Regression requires a numeric target. Choose a numeric target column or select classification.")
        valid_target = target_values.notna() & np.isfinite(pd.to_numeric(target_values, errors="coerce"))
    else:
        valid_target = target_values.notna()

    usable_indices = source_df.index[valid_target]
    usable_target = source_df.loc[usable_indices, req.target]
    if req.task == "classification" and usable_target.nunique(dropna=True) > 100:
        raise HTTPException(400, "Classification supports up to 100 target classes. Choose a categorical target or use regression for a continuous numeric target.")
    if usable_target.nunique(dropna=True) < 2 or len(usable_indices) < 10:
        raise HTTPException(400, "Need at least 10 usable rows and at least 2 target values.")

    max_training_rows = 1500
    sampled_for_memory = len(usable_indices) > max_training_rows
    if sampled_for_memory:
        stratify_sample = None
        if req.task == "classification":
            counts = usable_target.value_counts()
            if len(counts) <= 100 and counts.min() >= 2:
                stratify_sample = usable_target
        try:
            selected_indices, _ = train_test_split(
                np.asarray(usable_indices), train_size=max_training_rows,
                random_state=req.random_state, stratify=stratify_sample
            )
        except ValueError:
            selected_indices = np.random.default_rng(req.random_state).choice(
                np.asarray(usable_indices), size=max_training_rows, replace=False
            )
        df = source_df.loc[selected_indices].copy()
    else:
        df = source_df.loc[usable_indices].copy()
    if req.task == "regression":
        df[req.target] = pd.to_numeric(df[req.target], errors="coerce").astype(np.float64)

    if req.selected_features is None:
        input_feature_cols = [c for c in df.columns if c != req.target]
    else:
        unknown_features = [c for c in req.selected_features if c not in df.columns]
        if unknown_features:
            raise HTTPException(400, "Unknown input feature columns: " + ", ".join(unknown_features))
        input_feature_cols = [c for c in req.selected_features if c != req.target]
    if not input_feature_cols:
        raise HTTPException(400, "Select at least one input feature. The target column cannot also be an input feature.")

    X = df[input_feature_cols].copy().dropna(axis=1, how="all")
    input_feature_cols = X.columns.tolist()
    if not input_feature_cols:
        raise HTTPException(400, "All selected input features contain only missing values. Select other columns or preprocess the dataset.")
    if req.drop_high_missing:
        X = X.loc[:, X.isna().mean() <= .8]
        input_feature_cols = [c for c in input_feature_cols if c in X.columns]
    if X.shape[1] == 0:
        raise HTTPException(400, "No usable input features remain after missing-value filtering.")

    for c in X.select_dtypes(include=np.number).columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    for c in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            dt = pd.to_datetime(X[c], errors="coerce")
            X[c + "__year"] = dt.dt.year.astype(np.float32)
            X[c + "__month"] = dt.dt.month.astype(np.float32)
            X = X.drop(columns=[c])

    y = df[req.target].astype(str) if req.task == "classification" else df[req.target].astype(np.float64)
    test_rows = max(2, int(np.ceil(len(X) * req.test_size)))
    test_rows = min(test_rows, len(X) - 2)
    stratify = None
    if req.task == "classification":
        counts = y.value_counts()
        if counts.min() >= 2 and test_rows >= len(counts) and (len(X) - test_rows) >= len(counts):
            stratify = y
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_rows, random_state=req.random_state, stratify=stratify)
    except ValueError as exc:
        raise HTTPException(400, "Could not split the dataset. Increase the dataset size or adjust the test split. " + str(exc)) from exc

    pool = candidates(req.task)
    if req.mode == "auto":
        chosen = pool
    else:
        requested_names = req.model_names or [req.model_name]
        unknown_names = [name for name in requested_names if name not in pool]
        if unknown_names:
            raise HTTPException(400, "Unavailable model selection: " + ", ".join(unknown_names))
        chosen = {name: pool[name] for name in dict.fromkeys(requested_names)}
    if not chosen:
        raise HTTPException(400, "The selected model is not available for the chosen task. Refresh the model selection and try again.")

    leaderboard = []
    total_models = len(chosen)
    job_key = (user_id, assigned_model_id) if assigned_model_id else None
    job_started = time.monotonic()
    if job_key:
        update_training_record(user_id, assigned_model_id, {"total_models": total_models, "completed_models": 0, "progress_pct": 0, "current_model": None, "eta_seconds": None})
    best = None
    best_pipe = None
    best_predictions = []
    best_actual = []
    class_labels = list(pd.unique(y)) if req.task == "classification" else []

    for model_index, (name, estimator) in enumerate(chosen.items()):
        if job_key:
            elapsed = time.monotonic() - job_started
            avg = elapsed / model_index if model_index else None
            eta = int(avg * (total_models - model_index)) if avg is not None else None
            update_training_record(user_id, assigned_model_id, {"current_model": name, "completed_models": model_index, "total_models": total_models, "progress_pct": round(model_index / total_models * 100), "elapsed_seconds": int(elapsed), "eta_seconds": eta})
        headroom=available_memory_mb()
        if headroom is not None and headroom < 160:
            raise HTTPException(503, "Training paused before starting " + name + " because the server has less than 160 MB free memory. Select fewer models, reduce input features, or use a smaller dataset, then retry.")
        if cancel_event and cancel_event.is_set():
            raise HTTPException(409, "Training was cancelled.")
        pipeline = None
        predictions = None
        try:
            pipeline = make_pipeline(Xtr, estimator, req.scale_numeric, req.feature_selection, req.k_features, req.pca_components if req.use_pca else 0, req.task)
            pipeline.fit(Xtr, ytr)
            if cancel_event and cancel_event.is_set():
                raise HTTPException(409, "Training was cancelled.")
            predictions = pipeline.predict(Xte)
            if req.task == "classification":
                metrics = {
                    "accuracy": round(float(accuracy_score(yte, predictions)), 4),
                    "balanced_accuracy": round(float(balanced_accuracy_score(yte, predictions)), 4),
                    "precision_weighted": round(float(precision_score(yte, predictions, average="weighted", zero_division=0)), 4),
                    "recall_weighted": round(float(recall_score(yte, predictions, average="weighted", zero_division=0)), 4),
                    "f1_weighted": round(float(f1_score(yte, predictions, average="weighted", zero_division=0)), 4)
                }
                matrix = confusion_matrix(yte, predictions, labels=class_labels).tolist()
                evaluation = {"metrics": metrics, "confusion_matrix": {"labels": [str(v) for v in class_labels], "matrix": matrix}}
                score = metrics["f1_weighted"]
                score_name = "f1_weighted"
            else:
                mse = mean_squared_error(yte, predictions)
                r2 = float(r2_score(yte, predictions, force_finite=True))
                mae = float(mean_absolute_error(yte, predictions))
                metrics = {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(float(np.sqrt(mse)), 4), "mse": round(float(mse), 4)}
                evaluation = {"metrics": metrics}
                score = r2 if np.isfinite(r2) else -mae
                score_name = "r2" if np.isfinite(r2) else "negative_mae"

            leaderboard.append({"model_name": name, "status": "completed", "primary_metric": score_name, "primary_score": round(float(score), 4), **metrics})
            if best is None or score > best["primary_score"]:
                best = {"model_name": name, "metrics": metrics, "evaluation": evaluation, "primary_score": float(score)}
                best_pipe = pipeline
                best_predictions = [serial(value) for value in predictions]
                best_actual = [serial(value) for value in yte]
                pipeline = None
        except MemoryError as exc:
            raise HTTPException(503, "The server ran out of memory while fitting " + name + ". Select fewer models, reduce the dataset size, or use fewer input features, then retry.") from exc
        except Exception as exc:
            leaderboard.append({"model_name": name, "status": "failed", "primary_metric": None, "primary_score": None, "error": str(exc)})
        finally:
            if pipeline is not None:
                del pipeline
            if predictions is not None:
                del predictions
            del estimator
            gc.collect()
            if job_key:
                completed = model_index + 1
                elapsed = time.monotonic() - job_started
                avg = elapsed / completed
                update_training_record(user_id, assigned_model_id, {"completed_models": completed, "total_models": total_models, "progress_pct": round(completed / total_models * 100), "elapsed_seconds": int(elapsed), "eta_seconds": int(avg * (total_models - completed)), "current_model": None if completed == total_models else name})

    if cancel_event and cancel_event.is_set():
        raise HTTPException(409, "Training was cancelled.")
    successful = [row for row in leaderboard if row["status"] == "completed"]
    if not successful or best is None or best_pipe is None:
        details = "; ".join(row["model_name"] + ": " + row.get("error", "Unknown error") for row in leaderboard)
        raise HTTPException(400, "All selected models failed. " + details)
    leaderboard.sort(key=lambda row: row["primary_score"] if row["primary_score"] is not None else -float("inf"), reverse=True)

    _, model_dir, job_dir, _ = user_workspace(user_id)
    model_id = assigned_model_id or str(uuid.uuid4())
    created = datetime.now().isoformat()
    model_data = {"pipeline": best_pipe, "task": req.task, "target": req.target, "feature_cols": input_feature_cols, "model_name": best["model_name"], "dataset_id": req.dataset_id, "created_at": created}
    joblib.dump(model_data, model_dir / (model_id + ".joblib"), compress=3)

    job = {
        "job_id": model_id, "model_id": model_id, "dataset_id": req.dataset_id,
        "target": req.target, "task": req.task, "mode": req.mode,
        "best_model": best["model_name"], "created_at": created, "status": "completed",
        "train_rows": len(Xtr), "test_rows": len(Xte), "leaderboard": leaderboard,
        "best_evaluation": best["evaluation"], "request_config": req.model_dump()
    }
    (job_dir / (model_id + ".json")).write_text(json.dumps(job, indent=2), encoding="utf-8")

    warning_messages = [
        "Models were trained sequentially to reduce peak memory usage.",
        "Only the best model and its evaluation predictions were retained.",
        "Preprocessing was fitted on the training split."
    ]
    if sampled_for_memory:
        warning_messages.append("The dataset exceeded 1,500 usable rows. A reproducible sample of 1,500 rows was used to stay within the free server memory limit.")

    response = {
        "model_id": model_id, "task": req.task, "target": req.target, "mode": req.mode,
        "best_model": best["model_name"], "feature_columns": input_feature_cols,
        "train_rows": len(Xtr), "test_rows": len(Xte),
        "results": [{"model_name": best["model_name"], "metrics": best["metrics"], "evaluation": best["evaluation"], "predictions": best_predictions, "actual": best_actual}],
        "leaderboard": leaderboard, "best_evaluation": best["evaluation"],
        "warnings": warning_messages, "saved_locally": True
    }
    del source_df, df, X, y, Xtr, Xte, ytr, yte, best_pipe, model_data
    gc.collect()
    return response

@app.post("/api/predict")
async def predict(model_id: str = Form(...), file: UploadFile = File(...), include_actual_column: str = Form(default=""), user_id: str = Depends(current_user)):
    _, model_dir, _, export_dir = user_workspace(user_id)
    model_key = (user_id, model_id)
    if model_key in models:
        model_data = models[model_key]
    else:
        path = artifact_path(model_dir, model_id, ".joblib")
        if not path.exists():
            raise HTTPException(404, "Saved model not found.")
        model_data = joblib.load(path)
    content = await file.read()
    if len(content)>LIMIT: raise HTTPException(413,"CSV exceeds 100 MB.")
    try: df=clean(pd.read_csv(io.BytesIO(content),low_memory=False))
    except Exception as e: raise HTTPException(400,f"Could not parse test CSV: {e}")
    m=model_data; missing=[c for c in m["feature_cols"] if c not in df]
    if missing: raise HTTPException(400,"Missing required columns: "+", ".join(missing))
    X=df[m["feature_cols"]].copy()
    for c in X.select_dtypes(include=np.number).columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(np.float32)
    for c in X:
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            dt=pd.to_datetime(X[c],errors="coerce"); X[c+"__year"]=dt.dt.year; X[c+"__month"]=dt.dt.month; X=X.drop(columns=[c])
    pred=m["pipeline"].predict(X); out=df.copy(); out["Predicted_"+m["target"]]=pred
    eid=str(uuid.uuid4())
    (export_dir/f"{eid}.csv").write_text(out.to_csv(index=False),encoding="utf-8-sig")
    return {"export_id":eid,"rows":len(out),"predicted_column":"Predicted_"+m["target"],"preview":[{k:serial(v) for k,v in row.items()} for row in out.head(20).to_dict("records")],"download_url":f"/api/predictions/{eid}/download"}

@app.get("/api/predictions/{export_id}/download")
def download_predictions(export_id: str, user_id: str = Depends(current_user)):
    _, _, _, export_dir = user_workspace(user_id)
    path=artifact_path(export_dir, export_id, ".csv")
    if not path.exists(): raise HTTPException(404,"Prediction export not found.")
    return StreamingResponse(open(path,"rb"),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=predictions.csv"})

@app.get("/api/models/{model_id}/download-pkl")
def download_model_pkl(model_id: str, user_id: str = Depends(current_user)):
    _, model_dir, job_dir, _ = user_workspace(user_id)
    path = artifact_path(model_dir, model_id, ".joblib")
    if not path.is_file():
        raise HTTPException(404, "Saved model file not found. It may have been removed from the server workspace.")
    metadata_path = artifact_path(job_dir, model_id, ".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        metadata = {}
    safe = lambda value: (re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "model")).strip("-_")[:48] or "model")
    filename = "-".join([safe(metadata.get("best_model")), safe(metadata.get("task")), model_id[:8]]) + ".pkl"
    return StreamingResponse(open(path, "rb"), media_type="application/octet-stream", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.get("/api/models/{model_id}/evaluation.csv")
def download_evaluation_csv(model_id: str, user_id: str = Depends(current_user)):
    _, _, job_dir, _ = user_workspace(user_id)
    metadata_path = artifact_path(job_dir, model_id, ".json")
    if not metadata_path.is_file():
        raise HTTPException(404, "Experiment history was not found.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    leaderboard = metadata.get("leaderboard") or []
    if not leaderboard:
        raise HTTPException(404, "Evaluation results are not available for this experiment.")
    output = io.StringIO()
    pd.DataFrame(leaderboard).to_csv(output, index=False)
    filename = f"automl-evaluation-{model_id[:8]}.csv"
    return Response(output.getvalue().encode("utf-8-sig"), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{filename}"'})

@app.get("/api/models/{model_id}/download")
def download_model(model_id: str, user_id: str = Depends(current_user)):
    _, model_dir, job_dir, _ = user_workspace(user_id)
    path=artifact_path(model_dir, model_id, ".joblib")
    if not path.exists():
        raise HTTPException(404,"Saved model file not found. It may have been removed from the server workspace.")
    metadata_path=artifact_path(job_dir, model_id, ".json")
    metadata={}
    if metadata_path.exists():
        try:
            metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError,ValueError):
            metadata={}
    def filename_part(value: Any) -> str:
        cleaned=re.sub(r"[^A-Za-z0-9_-]+","-",str(value or "").strip()).strip("-_")
        return cleaned[:48] or "model"
    filename="-".join([
        filename_part(metadata.get("best_model")),
        filename_part(metadata.get("task")),
        filename_part(metadata.get("target")),
        model_id[:8]
    ])+".joblib"
    return StreamingResponse(
        open(path,"rb"),
        media_type="application/octet-stream",
        headers={"Content-Disposition":f'attachment; filename="{filename}"'}
    )

@app.get("/api/workspace")
def workspace(user_id: str = Depends(current_user)):
    data_dir, model_dir, _, _ = user_workspace(user_id)
    saved=[]
    for p in data_dir.glob("*.json"):
        try:
            m=json.loads(p.read_text(encoding="utf-8")); csv=artifact_path(data_dir,m["dataset_id"],".csv")
            if csv.exists(): m["size_bytes"]=csv.stat().st_size; saved.append(m)
        except Exception: pass
    return {"saved_datasets":sorted(saved,key=lambda x:x.get("saved_at",""),reverse=True),"saved_models":len(list(model_dir.glob("*.joblib")))}

@app.get("/api/workspace/jobs")
def saved_jobs(user_id: str = Depends(current_user)):
    _, model_dir, job_dir, _ = user_workspace(user_id)
    jobs=[]
    for p in job_dir.glob("*.json"):
        try:
            job=json.loads(p.read_text(encoding="utf-8"))
            job_id=job.get("job_id") or p.stem
            model_id=job.get("model_id") or job_id
            model_path=artifact_path(model_dir,model_id,".joblib")
            job["model_available"]=model_path.is_file()
            job["model_size_bytes"]=model_path.stat().st_size if model_path.is_file() else 0
            if job.get("status") in {"queued","running"}:
                job["status"]="interrupted"
                job["error"]="The server restarted before this experiment finished. Retry will start the saved configuration from the beginning."
                job["can_retry"]=bool(job.get("request_config"))
                p.write_text(json.dumps(job,indent=2,default=str),encoding="utf-8")
            jobs.append(job)
        except Exception:
            pass
    return {"jobs":sorted(jobs,key=lambda x:x.get("created_at",""),reverse=True)}

@app.post("/api/workspace/datasets/{dataset_id}/load")
def restore(dataset_id: str, user_id: str = Depends(current_user)):
    data_dir, _, _, _ = user_workspace(user_id)
    path=artifact_path(data_dir,dataset_id,".csv"); meta=artifact_path(data_dir,dataset_id,".json")
    if not path.exists(): raise HTTPException(404,"Saved dataset not found.")
    df=clean(pd.read_csv(path,low_memory=False)); datasets[(user_id,dataset_id)]=df
    name=json.loads(meta.read_text(encoding="utf-8")).get("filename",path.name) if meta.exists() else path.name
    return {"dataset_id":dataset_id,"filename":name,"saved_locally":True,**profile(df)}

@app.get("/api/datasets/{dataset_id}/eda/report.png")
def report(dataset_id: str, user_id: str = Depends(current_user)):
    import matplotlib.pyplot as plt
    df=get_df(dataset_id,user_id); cols=df.select_dtypes(include=np.number).columns[:8]
    fig,axes=plt.subplots(max(1,len(cols)),1,figsize=(10,max(4,2.5*len(cols)))); axes=np.atleast_1d(axes)
    if not len(cols): axes[0].text(.5,.5,"No numeric columns",ha="center"); axes[0].axis("off")
    for ax,c in zip(axes,cols): ax.hist(df[c].dropna(),bins=25); ax.set_title(c)
    output=io.BytesIO(); fig.tight_layout(); fig.savefig(output,format="png",dpi=140); plt.close(fig)
    return Response(output.getvalue(),media_type="image/png")
