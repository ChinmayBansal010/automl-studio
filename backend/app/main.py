from __future__ import annotations

import gc, hashlib, io, json, os, re, threading, uuid, warnings
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
from fastapi.responses import Response, StreamingResponse
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

def candidates(task: str) -> dict[str, Any]:
    if task == "classification":
        return {"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced"),
            "Decision Tree":DecisionTreeClassifier(random_state=42,class_weight="balanced"),
            "Random Forest":RandomForestClassifier(n_estimators=80,max_depth=16,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1,class_weight="balanced"),
            "Extra Trees":ExtraTreesClassifier(n_estimators=80,max_depth=16,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1,class_weight="balanced"),
            "Gradient Boosting":GradientBoostingClassifier(random_state=42),
            "Histogram Gradient Boosting":HistGradientBoostingClassifier(random_state=42),
            "AdaBoost":AdaBoostClassifier(random_state=42),"K-Nearest Neighbors":KNeighborsClassifier(),
            "Support Vector Machine":SVC(kernel="linear",class_weight="balanced",probability=False),"Naive Bayes":GaussianNB(),
            "Neural Network (MLP)":MLPClassifier(hidden_layer_sizes=(32,16),max_iter=150,early_stopping=True,random_state=42)}
    return {"Ridge Regression":Ridge(),"Decision Tree":DecisionTreeRegressor(random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=80,max_depth=16,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1),
        "Extra Trees":ExtraTreesRegressor(n_estimators=80,max_depth=16,min_samples_leaf=2,max_samples=0.8,random_state=42,n_jobs=1),
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
    if cats: parts.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=5,max_categories=20,dtype=np.float32))]),cats))
    steps: list[tuple[str,Any]] = [("prep",ColumnTransformer(parts,verbose_feature_names_out=False))]
    if selection == "kbest": steps.append(("select",SelectKBest(f_classif if task=="classification" else f_regression,k=max(1,min(k,len(X.columns))))))
    if selection == "variance": steps.append(("select",VarianceThreshold()))
    if pca: steps.append(("pca",PCA(n_components=max(1,min(pca,len(X.columns))),random_state=42)))
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
    return {"dataset_id":did,"filename":file.filename,"saved_locally":True,**profile(df)}

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
    df=get_df(dataset_id, user_id).copy(); cols=[c for c in req.columns if c in df]
    op=req.operation
    if op=="drop_duplicates": df=df.drop_duplicates()
    elif op=="drop_missing_rows": df=df.dropna(subset=cols or None)
    elif op=="drop_columns":
        if not cols: raise HTTPException(400,"Select columns to drop.")
        df=df.drop(columns=cols)
    elif op=="fill_missing":
        selected=cols or df.columns.tolist()
        if req.strategy=="drop": df=df.dropna(subset=selected)
        else:
            for c in selected:
                if req.strategy=="mean" and pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(df[c].mean())
                elif req.strategy=="median" and pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(df[c].median())
                elif req.strategy=="mode" and not df[c].mode().empty: df[c]=df[c].fillna(df[c].mode().iloc[0])
                elif req.strategy=="constant": df[c]=df[c].fillna(req.value or "Unknown")
                elif req.strategy=="zero" and pd.api.types.is_numeric_dtype(df[c]): df[c]=df[c].fillna(0)
                elif req.strategy=="ffill": df[c]=df[c].ffill()
                elif req.strategy=="bfill": df[c]=df[c].bfill()
    elif op=="drop_high_missing": df=df.drop(columns=[c for c in df if df[c].isna().mean()>req.threshold])
    elif op in ("drop_outliers","clip_outliers"):
        for c in cols or df.select_dtypes(include=np.number).columns:
            if not pd.api.types.is_numeric_dtype(df[c]): continue
            q1,q3=df[c].quantile(.25),df[c].quantile(.75); iqr=q3-q1
            if op=="clip_outliers": df[c]=df[c].clip(q1-1.5*iqr,q3+1.5*iqr)
            else: df=df[df[c].between(q1-1.5*iqr,q3+1.5*iqr)|df[c].isna()]
    elif op=="convert_numeric":
        for c in cols: df[c]=pd.to_numeric(df[c],errors="coerce")
    elif op=="convert_datetime":
        for c in cols: df[c]=pd.to_datetime(df[c],errors="coerce")
    elif op=="rename_column":
        if len(cols)!=1 or not req.new_name.strip(): raise HTTPException(400,"Select one column and enter its new name.")
        if req.new_name in df and req.new_name!=cols[0]: raise HTTPException(400,"Column name already exists.")
        df=df.rename(columns={cols[0]:req.new_name.strip()})
    elif op=="sample_rows": df=df.sample(n=min(req.sample_size,len(df)),random_state=42)
    elif op=="reset_index": df=df.reset_index(drop=True)
    else: raise HTTPException(400,"Unknown preprocessing operation.")
    if df.empty: raise HTTPException(400,"Operation would create an empty dataset.")
    data_dir, _, _, _ = user_workspace(user_id)
    datasets[(user_id, dataset_id)]=df; df.to_csv(artifact_path(data_dir, dataset_id, ".csv"),index=False)
    return {"dataset_id":dataset_id,"saved_locally":True,**profile(df)}

@app.post("/api/datasets/{dataset_id}/feature-engineering")
def feature_engineering(dataset_id: str, payload: dict[str,Any], user_id: str = Depends(current_user)):
    df=get_df(dataset_id, user_id).copy(); op=payload.get("operation"); cols=[c for c in payload.get("columns",[]) if c in df]
    if op=="one_hot_encode": df=pd.get_dummies(df,columns=cols,dummy_na=bool(payload.get("dummy_na",False)),dtype=int)
    elif op in ("standardize","minmax","log_transform","polynomial"):
        for c in cols:
            if not pd.api.types.is_numeric_dtype(df[c]): continue
            if op=="standardize": df[c]=(df[c]-df[c].mean())/(df[c].std() or 1)
            elif op=="minmax": df[c]=(df[c]-df[c].min())/((df[c].max()-df[c].min()) or 1)
            elif op=="log_transform": df[c]=np.log1p(df[c].clip(lower=0))
            else: df[f"{c}__squared"]=df[c]**2
    elif op=="pca":
        cols=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(cols)<2: raise HTTPException(400,"PCA requires at least two numeric columns.")
        x=df[cols].apply(pd.to_numeric,errors="coerce"); x=x.fillna(x.median()).fillna(0)
        n=min(int(payload.get("components",2)),len(cols),len(df))
        z=PCA(n_components=n,random_state=42).fit_transform(StandardScaler().fit_transform(x))
        df=df.drop(columns=cols)
        for i in range(n): df[f"PC{i+1}"]=z[:,i]
    else: raise HTTPException(400,"Unknown feature engineering operation.")
    data_dir, _, _, _ = user_workspace(user_id)
    datasets[(user_id, dataset_id)]=df; df.to_csv(artifact_path(data_dir, dataset_id, ".csv"),index=False)
    return {"dataset_id":dataset_id,"saved_locally":True,**profile(df)}

@app.post("/api/datasets/{dataset_id}/chart")
def chart(dataset_id: str, payload: dict[str,Any], user_id: str = Depends(current_user)):
    df=get_df(dataset_id, user_id)
    kind=payload.get("chart_type","bar")
    x=payload.get("x")
    y=payload.get("y")
    rows=max(1,min(int(payload.get("rows",20)),500))
    agg=payload.get("aggregation","mean")
    if kind=="heatmap":
        numeric=df.select_dtypes(include=np.number).iloc[:,:12]
        corr=numeric.corr().round(3).replace({np.nan:None})
        data=[{"x":str(i),**{str(k):serial(v) for k,v in r.items()}} for i,r in corr.iterrows()]
        return {"chart_type":kind,"x":None,"y":None,"rows":rows,"columns":max(1,min(int(payload.get("columns",1)),6)),"data":data}
    if kind=="histogram":
        column=x if x in df else y
        if column not in df or not pd.api.types.is_numeric_dtype(df[column]):
            raise HTTPException(400,"Choose a numeric column for the histogram.")
        vals=pd.to_numeric(df[column],errors="coerce").dropna()
        counts,edges=np.histogram(vals,bins=min(rows,60)) if len(vals) else ([],[])
        data=[{"x":f"{edges[i]:.4g}–{edges[i+1]:.4g}","value":int(counts[i])} for i in range(len(counts))]
        return {"chart_type":kind,"x":column,"y":None,"rows":rows,"columns":max(1,min(int(payload.get("columns",1)),6)),"data":data}
    if x not in df:
        raise HTTPException(400,"Choose a valid category or X-axis column.")
    if kind in ("pie","count"):
        data=[{"x":str(k),"value":int(v)} for k,v in df[x].astype("string").fillna("(Missing)").value_counts().head(rows).items()]
    elif kind in ("line","area","scatter"):
        if y not in df:
            raise HTTPException(400,"Choose a valid Y-axis column.")
        if kind=="scatter":
            if not pd.api.types.is_numeric_dtype(df[x]) or not pd.api.types.is_numeric_dtype(df[y]):
                raise HTTPException(400,"Scatter plots require numeric X and Y columns.")
            data=[{"x":serial(a),"value":serial(b)} for a,b in zip(df[x].head(rows),df[y].head(rows)) if pd.notna(a) and pd.notna(b)]
        else:
            if not pd.api.types.is_numeric_dtype(df[y]):
                raise HTTPException(400,"Choose a numeric Y-axis column.")
            grouped=df.groupby(x,dropna=False)[y]
            values={"sum":grouped.sum(),"count":grouped.count(),"median":grouped.median(),"min":grouped.min(),"max":grouped.max()}.get(agg,grouped.mean()).head(rows)
            data=[{"x":str(k),"value":serial(v)} for k,v in values.items()]
    else:
        if y in df and pd.api.types.is_numeric_dtype(df[y]):
            grouped=df.groupby(x,dropna=False)[y]
            values={"sum":grouped.sum(),"count":grouped.count(),"median":grouped.median(),"min":grouped.min(),"max":grouped.max()}.get(agg,grouped.mean()).head(rows)
            data=[{"x":str(k),"value":serial(v)} for k,v in values.items()]
        else:
            data=[{"x":str(k),"value":int(v)} for k,v in df[x].astype("string").fillna("(Missing)").value_counts().head(rows).items()]
    return {"chart_type":kind,"x":x,"y":y if kind not in ("pie","count","histogram") else None,"rows":rows,"columns":max(1,min(int(payload.get("columns",1)),6)),"data":data}

class Train(BaseModel):
    dataset_id: str
    target: str
    task: str = Field(pattern="^(classification|regression)$")
    mode: str = Field(default="auto",pattern="^(auto|manual)$")
    model_name: str = "Random Forest"
    test_size: float = Field(default=.2,ge=.1,le=.5)
    random_state: int = 42
    scale_numeric: bool = True
    drop_high_missing: bool = False
    selected_features: list[str] | None = None
    feature_selection: str = Field(default="none",pattern="^(none|kbest|variance)$")
    k_features: int = Field(default=10,ge=1,le=500)
    use_pca: bool = False
    pca_components: int = Field(default=0,ge=0,le=500)

TRAINING_USERS: set[str] = set()
TRAINING_USERS_LOCK = threading.Lock()

@app.post("/api/train")
def train(req: Train, user_id: str = Depends(current_user)):
    with TRAINING_USERS_LOCK:
        if user_id in TRAINING_USERS:
            raise HTTPException(409, "A training job is already running for this account. Wait for it to finish before starting another.")
        TRAINING_USERS.add(user_id)
    try:
        return run_training(req, user_id)
    finally:
        with TRAINING_USERS_LOCK:
            TRAINING_USERS.discard(user_id)

def run_training(req: Train, user_id: str):
    source_df=get_df(req.dataset_id,user_id)
    if req.target not in source_df.columns: raise HTTPException(400,"Target column not found.")
    df=source_df.dropna(subset=[req.target]).copy()
    sampled_for_memory=len(df)>8000
    if sampled_for_memory:
        sample_stratify=df[req.target] if req.task=="classification" and df[req.target].value_counts().min()>=2 else None
        df,_=train_test_split(df,train_size=8000,random_state=req.random_state,stratify=sample_stratify)
    if req.task=="classification" and df[req.target].nunique(dropna=True)>100: raise HTTPException(400,"Target has more than 100 classes. Choose another target or use regression for a continuous numeric target.")
    if req.task=="regression" and not pd.api.types.is_numeric_dtype(df[req.target]): raise HTTPException(400,"Regression requires a numeric target.")
    if df[req.target].nunique()<2 or len(df)<10: raise HTTPException(400,"Need at least 10 rows and 2 target values.")
    cols=[c for c in (req.selected_features or df.columns.tolist()) if c in df and c!=req.target]
    X=df[cols].dropna(axis=1,how="all").copy()
    y=df[req.target].astype(str) if req.task=="classification" else df[req.target]
    for c in X.select_dtypes(include=np.number).columns: X[c]=pd.to_numeric(X[c],errors="coerce").astype(np.float32)
    for c in list(X.columns):
        if pd.api.types.is_datetime64_any_dtype(X[c]):
            dt=pd.to_datetime(X[c],errors="coerce")
            X[c+"__year"]=dt.dt.year.astype(np.float32); X[c+"__month"]=dt.dt.month.astype(np.float32); X=X.drop(columns=[c])
    if req.drop_high_missing: X=X.loc[:,X.isna().mean()<=.8]
    if X.shape[1]==0: raise HTTPException(400,"No usable input features.")
    strat=y if req.task=="classification" and y.value_counts().min()>=2 else None
    try: Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=req.test_size,random_state=req.random_state,stratify=strat)
    except ValueError as e: raise HTTPException(400,f"Could not split data: {e}") from e
    pool=candidates(req.task)
    chosen=pool if req.mode=="auto" else {req.model_name:pool[req.model_name]} if req.model_name in pool else {}
    if not chosen: raise HTTPException(400,"Selected model is not available for this task.")
    leaderboard=[]; best=None; best_pipe=None; best_predictions=[]; best_actual=[]
    for name,est in chosen.items():
        pipe=None; pred=None
        try:
            pipe=make_pipeline(Xtr,est,req.scale_numeric,req.feature_selection,req.k_features,req.pca_components if req.use_pca else 0,req.task)
            pipe.fit(Xtr,ytr); pred=pipe.predict(Xte)
            if req.task=="classification":
                metrics={"accuracy":round(float(accuracy_score(yte,pred)),4),"balanced_accuracy":round(float(balanced_accuracy_score(yte,pred)),4),"precision_weighted":round(float(precision_score(yte,pred,average="weighted",zero_division=0)),4),"recall_weighted":round(float(recall_score(yte,pred,average="weighted",zero_division=0)),4),"f1_weighted":round(float(f1_score(yte,pred,average="weighted",zero_division=0)),4)}
                labels=list(pd.Series(yte).unique()); evaluation={"metrics":metrics,"confusion_matrix":{"labels":[str(v) for v in labels],"matrix":confusion_matrix(yte,pred,labels=labels).tolist()}}; score=metrics["f1_weighted"]
            else:
                mse=mean_squared_error(yte,pred); metrics={"r2":round(float(r2_score(yte,pred)),4),"mae":round(float(mean_absolute_error(yte,pred)),4),"rmse":round(float(np.sqrt(mse)),4),"mse":round(float(mse),4)}
                evaluation={"metrics":metrics}; score=metrics["r2"]
            leaderboard.append({"model_name":name,"metrics":metrics,"primary_score":score})
            if best is None or not np.isfinite(best["primary_score"]) or (np.isfinite(score) and score>best["primary_score"]):
                best={"model_name":name,"metrics":metrics,"evaluation":evaluation,"primary_score":score}
                best_pipe=pipe; best_predictions=[serial(v) for v in pred]; best_actual=[serial(v) for v in yte]; pipe=None
        except Exception as e:
            leaderboard.append({"model_name":name,"error":str(e),"metrics":{},"primary_score":None})
        finally:
            if pipe is not None: del pipe
            if pred is not None: del pred
            del est
            gc.collect()
    successful=[r for r in leaderboard if "error" not in r]
    if not successful or best is None or best_pipe is None:
        raise HTTPException(400,"All candidate models failed: "+"; ".join(r["model_name"]+": "+r.get("error","") for r in leaderboard))
    leaderboard=sorted(successful,key=lambda r:r["primary_score"] if r["primary_score"] is not None else -999,reverse=True)
    _,model_dir,job_dir,_=user_workspace(user_id); mid=str(uuid.uuid4()); created=datetime.now().isoformat()
    model_data={"pipeline":best_pipe,"task":req.task,"target":req.target,"feature_cols":cols,"model_name":best["model_name"],"dataset_id":req.dataset_id,"created_at":created}
    models[(user_id,mid)]=model_data; joblib.dump(model_data,model_dir/f"{mid}.joblib",compress=3)
    job={"job_id":mid,"model_id":mid,"dataset_id":req.dataset_id,"target":req.target,"task":req.task,"mode":req.mode,"best_model":best["model_name"],"created_at":created,"train_rows":len(Xtr),"test_rows":len(Xte),"leaderboard":[{"model_name":r["model_name"],"primary_score":r["primary_score"],**r["metrics"]} for r in leaderboard]}
    (job_dir/f"{mid}.json").write_text(json.dumps(job,indent=2),encoding="utf-8")
    response={"model_id":mid,"task":req.task,"target":req.target,"mode":req.mode,"best_model":best["model_name"],"feature_columns":cols,"train_rows":len(Xtr),"test_rows":len(Xte),"results":[{"model_name":best["model_name"],"metrics":best["metrics"],"evaluation":best["evaluation"],"predictions":best_predictions,"actual":best_actual}],"leaderboard":job["leaderboard"],"best_evaluation":best["evaluation"],"warnings":["Models are trained sequentially to reduce peak memory usage.","Only the best model predictions are retained.","Preprocessing is fitted on the training split.","Holdout metrics are estimates; repeated tuning can overfit the holdout."].concat(sampled_for_memory ? ["Dataset exceeded 8,000 rows; a stratified random sample of 8,000 rows was used to stay within the free server memory limit."] : []),"saved_locally":True}
    del source_df,df,X,y,Xtr,Xte,ytr,yte,best_pipe,model_data
    gc.collect()
    return response

@app.post("/api/predict")
async def predict(model_id: str = Form(...), file: UploadFile = File(...), include_actual_column: str = Form(default=""), user_id: str = Depends(current_user)):
    _, model_dir, _, export_dir = user_workspace(user_id)
    model_key = (user_id, model_id)
    if model_key not in models:
        path=artifact_path(model_dir, model_id, ".joblib")
        if path.exists(): models[model_key]=joblib.load(path)
        else: raise HTTPException(404,"Saved model not found.")
    content = await file.read()
    if len(content)>LIMIT: raise HTTPException(413,"CSV exceeds 100 MB.")
    try: df=clean(pd.read_csv(io.BytesIO(content),low_memory=False))
    except Exception as e: raise HTTPException(400,f"Could not parse test CSV: {e}")
    m=models[model_key]; missing=[c for c in m["feature_cols"] if c not in df]
    if missing: raise HTTPException(400,"Missing required columns: "+", ".join(missing))
    X=df[m["feature_cols"]].copy()
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

@app.get("/api/models/{model_id}/download")
def download_model(model_id: str, user_id: str = Depends(current_user)):
    _, model_dir, _, _ = user_workspace(user_id)
    path=artifact_path(model_dir, model_id, ".joblib")
    if not path.exists(): raise HTTPException(404,"Model file not found.")
    return StreamingResponse(open(path,"rb"),media_type="application/octet-stream",headers={"Content-Disposition":f"attachment; filename={model_id}.joblib"})

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
    _, _, job_dir, _ = user_workspace(user_id)
    jobs=[]
    for p in job_dir.glob("*.json"):
        try: jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception: pass
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
