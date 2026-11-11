"""
Regression analysis server for n8n.

n8n (Google Sheets / Supabase node) -> HTTP Request node -> POST /webhook/regression
-> JSON response with the fitted regression model (metrics, coefficients, predictions).

Optionally pass `callback_url` (an n8n Webhook node URL) to run asynchronously:
the server replies 202 immediately and POSTs the result to that URL when done.

One endpoint, three tasks (set `task` in the JSON body; default is "regression"):
  - "regression": fit a regression model (unchanged behaviour)
  - "eda":        exploratory data analysis (summary stats, histograms, correlations)
  - "cluster":    K-means clustering (auto-selects k by silhouette if `n_clusters` is omitted)
"""
import logging
import math
import os
import re
from typing import Any, Literal, Optional

import httpx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regression-server")

API_KEY = os.getenv("API_KEY")  # optional; if set, clients must send X-API-Key
MIN_ROWS = 5

app = FastAPI(title="n8n Regression Server", version="1.1.0")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def check_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #
class RegressionRequest(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="Rows from Google Sheets / Supabase")
    target: Optional[str] = Field(None, description="Column to predict (default: last numeric column)")
    features: Optional[list[str]] = Field(None, description="Predictor columns (default: all others)")
    model_type: Literal["linear", "ridge", "lasso", "polynomial", "random_forest"] = "linear"
    alpha: float = Field(1.0, gt=0, description="Regularisation for ridge/lasso")
    degree: int = Field(2, ge=2, le=5, description="Degree for polynomial regression")
    test_size: float = Field(0.2, ge=0, lt=0.9)
    cv_folds: int = Field(5, ge=0, le=20, description="0 disables cross-validation")
    random_state: int = 42
    predict: Optional[list[dict[str, Any]]] = Field(None, description="New rows to predict")
    callback_url: Optional[str] = Field(None, description="n8n Webhook URL for async result")
    return_fitted: bool = Field(False, description="Include fitted values and the input row index of each")
    task: Literal["regression", "eda", "cluster"] = Field("regression", description="Which analysis to run")
    n_clusters: Optional[int] = Field(None, ge=2, le=20, description="K for K-means (omit to auto-select)")
    max_clusters: int = Field(8, ge=2, le=20, description="Largest K tried when auto-selecting")
    bins: int = Field(10, ge=3, le=50, description="Histogram bins for EDA")

    @field_validator("data", mode="before")
    @classmethod
    def unwrap_n8n_items(cls, v):
        # n8n items look like [{"json": {...}}, ...] when $input.all() is sent raw
        if isinstance(v, list) and v and all(isinstance(r, dict) and isinstance(r.get("json"), dict) for r in v):
            return [r["json"] for r in v]
        return v


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _f(x: Any) -> Optional[float]:
    """JSON-safe float."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Google Sheets returns everything as strings: blanks -> NaN, numeric-looking text -> numbers."""
    df = df.replace(r"^\s*$", np.nan, regex=True)
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)
        elif df[c].dtype == object:
            non_null = df[c].notna().sum()
            if non_null == 0:
                continue
            conv = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")
            if conv.notna().sum() >= 0.9 * non_null:
                df[c] = conv
    return df


def _build_pipeline(model_type: str, num_cols: list, cat_cols: list, alpha: float, degree: int, seed: int) -> Pipeline:
    num_steps: list = [("impute", SimpleImputer(strategy="median"))]
    if model_type == "polynomial":
        num_steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    if model_type in {"ridge", "lasso", "polynomial"}:
        num_steps.append(("scale", StandardScaler()))

    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                drop=None if model_type == "random_forest" else "first",
            ),
        ),
    ]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline(num_steps), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))
    pre = ColumnTransformer(transformers, verbose_feature_names_out=False)

    model = {
        "linear": LinearRegression(),
        "polynomial": LinearRegression(),
        "ridge": Ridge(alpha=alpha),
        "lasso": Lasso(alpha=alpha, max_iter=20000),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
    }[model_type]
    return Pipeline([("pre", pre), ("model", model)])


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": _f(r2_score(y_true, y_pred)),
        "rmse": _f(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": _f(mean_absolute_error(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Core analysis
# --------------------------------------------------------------------------- #
def run_regression(req: RegressionRequest) -> dict:
    warnings: list[str] = []
    if not req.data:
        raise ValueError("`data` is empty")

    df = _coerce(pd.DataFrame(req.data))
    n_received = len(df)

    # ---- target -----------------------------------------------------------
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    target = req.target
    if target is None:
        if not numeric_cols:
            raise ValueError("No numeric column found to use as target")
        target = numeric_cols[-1]
        warnings.append(f"`target` not provided; using last numeric column '{target}'")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found. Columns: {list(df.columns)}")
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(subset=[target])
    kept_idx = df.index.tolist()  # positions in the submitted data
    df = df.reset_index(drop=True)

    # ---- features ---------------------------------------------------------
    feats = req.features or [c for c in df.columns if c != target]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Features not found: {missing}")
    if target in feats:
        raise ValueError("Target cannot also be a feature")

    X, y = df[feats].copy(), df[target].astype(float)
    dropped: dict[str, str] = {}
    for c in list(X.columns):
        nun = X[c].nunique(dropna=True)
        is_num = pd.api.types.is_numeric_dtype(X[c])
        if X[c].isna().all():
            dropped[c] = "all values missing"
        elif nun <= 1:
            dropped[c] = "constant"
        elif not is_num and (nun > 50 or nun == len(X)):
            dropped[c] = "high-cardinality / ID-like text"
    X = X.drop(columns=list(dropped))
    if X.shape[1] == 0:
        raise ValueError(f"No usable feature columns left. Dropped: {dropped}")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    n = len(X)
    if n < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows with a valid target; got {n}")

    pipe = _build_pipeline(req.model_type, num_cols, cat_cols, req.alpha, req.degree, req.random_state)

    # ---- evaluation (train/test split) -----------------------------------
    result_metrics: dict[str, Any] = {}
    if req.test_size > 0 and n >= 20:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state)
        eval_pipe = clone(pipe).fit(X_tr, y_tr)
        result_metrics["train"] = _metrics(y_tr, eval_pipe.predict(X_tr))
        result_metrics["test"] = _metrics(y_te, eval_pipe.predict(X_te))
        n_train, n_test = len(X_tr), len(X_te)
    else:
        if req.test_size > 0:
            warnings.append("Fewer than 20 rows: skipped train/test split, metrics are in-sample")
        n_train, n_test = n, 0
        result_metrics["train"] = None

    if req.cv_folds >= 2 and n >= req.cv_folds * 2:
        try:
            cv = cross_val_score(clone(pipe), X, y, cv=req.cv_folds, scoring="r2")
            result_metrics["cv_r2_mean"] = _f(cv.mean())
            result_metrics["cv_r2_std"] = _f(cv.std())
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Cross-validation failed: {e}")

    # ---- final model on all data -----------------------------------------
    pipe.fit(X, y)
    fitted = pipe.predict(X)
    result_metrics["full_fit"] = _metrics(y, fitted)
    names = [str(s) for s in pipe.named_steps["pre"].get_feature_names_out()]
    model = pipe.named_steps["model"]

    coefs: list[dict[str, Any]] = []
    intercept = None
    equation = None
    if req.model_type == "random_forest":
        coefs = [
            {"feature": nm, "importance": _f(v)}
            for nm, v in sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])
        ]
    else:
        intercept = _f(model.intercept_)
        coefs = [{"feature": nm, "coefficient": _f(v)} for nm, v in zip(names, model.coef_)]
        if req.model_type == "lasso":
            coefs.sort(key=lambda d: -abs(d["coefficient"] or 0))
        if req.model_type in {"ridge", "lasso", "polynomial"}:
            warnings.append("Coefficients refer to standardized features")

        # Statistical significance (OLS) for plain linear regression
        if req.model_type == "linear":
            try:
                Xt = pipe.named_steps["pre"].transform(X)
                if n > Xt.shape[1] + 1:
                    ols = sm.OLS(y.values, sm.add_constant(Xt, has_constant="add")).fit()
                    ci = ols.conf_int()
                    for i, c in enumerate(coefs, start=1):
                        c["std_err"] = _f(ols.bse[i])
                        c["p_value"] = _f(ols.pvalues[i])
                        c["ci_95"] = [_f(ci[i][0]), _f(ci[i][1])]
                    result_metrics["adj_r2"] = _f(ols.rsquared_adj)
                    result_metrics["f_pvalue"] = _f(ols.f_pvalue)
                else:
                    warnings.append("Too few rows for p-values (need rows > features + 1)")
            except Exception as e:  # noqa: BLE001
                warnings.append(f"OLS statistics unavailable: {e}")
            terms = " ".join(f"{'+' if (c['coefficient'] or 0) >= 0 else '-'} {abs(c['coefficient'] or 0):.4g}*{c['feature']}" for c in coefs)
            equation = f"{target} = {intercept:.4g} {terms}"

    # ---- predictions for new rows ----------------------------------------
    predictions = None
    if req.predict:
        new = _coerce(pd.DataFrame(req.predict))
        for c in X.columns:
            if c not in new.columns:
                new[c] = np.nan
        new = new[list(X.columns)]
        for c in num_cols:
            new[c] = pd.to_numeric(new[c], errors="coerce")
        predictions = [_f(v) for v in pipe.predict(new)]

    resid = y.values - fitted
    return {
        "status": "ok",
        "model_type": req.model_type,
        "target": target,
        "features": {"numeric": num_cols, "categorical": cat_cols, "dropped": dropped},
        "rows": {"received": n_received, "used": n, "train": n_train, "test": n_test},
        "metrics": result_metrics,
        "intercept": intercept,
        "coefficients": coefs,
        "equation": equation,
        "residuals": {
            "mean": _f(resid.mean()),
            "std": _f(resid.std(ddof=1)) if n > 1 else None,
            "min": _f(resid.min()),
            "max": _f(resid.max()),
        },
        "predictions": predictions,
        "fitted": [_f(v) for v in fitted] if req.return_fitted else None,
        "row_index": kept_idx if req.return_fitted else None,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
ID_NAME = re.compile(r"(^|[_\s])(id|index|idx)$|^unnamed", re.IGNORECASE)


def run_eda(req: RegressionRequest) -> dict:
    warnings: list[str] = []
    if not req.data:
        raise ValueError("`data` is empty")

    df = _coerce(pd.DataFrame(req.data))
    n = len(df)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    numeric: list[dict[str, Any]] = []
    for c in num_cols:
        s = df[c].dropna()
        entry: dict[str, Any] = {
            "name": str(c),
            "count": int(len(s)),
            "missing": int(n - len(s)),
            "missing_pct": _f(100 * (n - len(s)) / n),
            "unique": int(s.nunique()),
        }
        if s.empty:
            warnings.append(f"'{c}' has no values")
            numeric.append(entry)
            continue
        q1, med, q3 = (float(v) for v in s.quantile([0.25, 0.5, 0.75]))
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        counts, edges = np.histogram(s, bins=req.bins)
        entry.update(
            {
                "mean": _f(s.mean()),
                "std": _f(s.std(ddof=1)) if len(s) > 1 else None,
                "min": _f(s.min()),
                "q1": _f(q1),
                "median": _f(med),
                "q3": _f(q3),
                "max": _f(s.max()),
                "skew": _f(s.skew()) if len(s) > 2 else None,
                "outliers_iqr": outliers,
                "histogram": {"edges": [_f(e) for e in edges], "counts": [int(v) for v in counts]},
            }
        )
        if entry["missing_pct"] and entry["missing_pct"] > 30:
            warnings.append(f"'{c}' is {entry['missing_pct']:.0f}% missing")
        if entry["unique"] <= 1:
            warnings.append(f"'{c}' is constant")
        if ID_NAME.search(str(c)) and entry["unique"] == entry["count"]:
            warnings.append(f"'{c}' looks like an ID/index; exclude it from predictors and clustering")
        numeric.append(entry)

    categorical: list[dict[str, Any]] = []
    for c in cat_cols:
        s = df[c].dropna().astype(str)
        top = s.value_counts().head(8)
        categorical.append(
            {
                "name": str(c),
                "count": int(len(s)),
                "missing": int(n - len(s)),
                "missing_pct": _f(100 * (n - len(s)) / n),
                "unique": int(s.nunique()),
                "top": [{"value": str(k), "count": int(v)} for k, v in top.items()],
            }
        )
        if n > 1 and len(s) and s.nunique() == len(s):
            warnings.append(f"'{c}' is unique per row (ID-like); not useful as a predictor")

    # ---- correlations (Pearson, pairwise-complete) -----------------------
    usable = [c for c in num_cols if df[c].nunique(dropna=True) > 1]
    if len(usable) > 20:
        warnings.append(f"Correlation matrix limited to the first 20 of {len(usable)} numeric columns")
        usable = usable[:20]
    correlation = None
    top_pairs: list[dict[str, Any]] = []
    if len(usable) >= 2:
        corr = df[usable].corr()
        correlation = {
            "columns": [str(c) for c in usable],
            "matrix": [[_f(v) for v in row] for row in corr.values],
        }
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                r = _f(corr.iloc[i, j])
                if r is not None:
                    top_pairs.append({"a": str(usable[i]), "b": str(usable[j]), "r": r})
        top_pairs.sort(key=lambda d: -abs(d["r"]))
        top_pairs = top_pairs[:10]

    target_corr = None
    if req.target and req.target in usable:
        target_corr = sorted(
            (
                {"feature": str(c), "r": _f(df[c].corr(df[req.target]))}
                for c in usable
                if c != req.target and _f(df[c].corr(df[req.target])) is not None
            ),
            key=lambda d: -abs(d["r"]),
        )
    elif req.target:
        warnings.append(f"`target` '{req.target}' is not a usable numeric column; skipped target correlations")

    return {
        "status": "ok",
        "task": "eda",
        "rows": n,
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric": numeric,
        "categorical": categorical,
        "correlation": correlation,
        "top_correlations": top_pairs,
        "target_correlations": target_corr,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# K-means clustering
# --------------------------------------------------------------------------- #
def _safe_silhouette(Z: np.ndarray, labels: np.ndarray, seed: int) -> Optional[float]:
    n_labels = len(set(labels.tolist()))
    if n_labels < 2 or n_labels >= len(Z):
        return None
    try:
        return _f(silhouette_score(Z, labels, sample_size=min(len(Z), 2000), random_state=seed))
    except Exception:  # noqa: BLE001
        return None


def run_cluster(req: RegressionRequest) -> dict:
    warnings: list[str] = []
    if not req.data:
        raise ValueError("`data` is empty")

    df = _coerce(pd.DataFrame(req.data))
    n_received = len(df)

    if req.features:
        missing = [c for c in req.features if c not in df.columns]
        if missing:
            raise ValueError(f"Features not found: {missing}")
        cols = list(req.features)
    else:
        cols = df.select_dtypes(include="number").columns.tolist()
        if req.target and req.target in cols and len(cols) > 1:
            warnings.append(f"`target` '{req.target}' is included as a clustering feature; pass `features` to choose")

    if not cols:
        raise ValueError("No numeric columns found to cluster")
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    dropped: dict[str, str] = {}
    for c in list(X.columns):
        s = X[c]
        if s.isna().all():
            dropped[c] = "no numeric values"
        elif s.nunique(dropna=True) <= 1:
            dropped[c] = "constant"
        elif not req.features and ID_NAME.search(str(c)) and s.nunique() == s.notna().sum():
            dropped[c] = "ID-like column"
    X = X.drop(columns=list(dropped))
    if X.shape[1] == 0:
        raise ValueError(f"No usable numeric columns to cluster. Dropped: {dropped}")

    keep = X.notna().any(axis=1)
    kept_idx = X.index[keep].tolist()  # positions in the submitted data
    X = X[keep].reset_index(drop=True)
    n = len(X)
    if n < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows; got {n}")

    Xi = SimpleImputer(strategy="median").fit_transform(X)
    scaler = StandardScaler().fit(Xi)
    Z = scaler.transform(Xi)
    warnings.append("Features were standardized before clustering")

    # ---- choose k --------------------------------------------------------
    k_given = req.n_clusters
    if k_given is not None and k_given >= n:
        raise ValueError(f"n_clusters ({k_given}) must be smaller than the number of rows ({n})")
    k_max = min(req.max_clusters, n - 1)
    ks = set(range(2, k_max + 1))
    if k_given:
        ks.add(k_given)
    k_search: list[dict[str, Any]] = []
    for kk in sorted(ks):
        km_try = KMeans(n_clusters=kk, n_init=5, random_state=req.random_state).fit(Z)
        k_search.append(
            {
                "k": kk,
                "inertia": _f(km_try.inertia_),
                "silhouette": _safe_silhouette(Z, km_try.labels_, req.random_state),
            }
        )
    if k_given:
        k, how = k_given, "user"
    else:
        scored = [r for r in k_search if r["silhouette"] is not None]
        if not scored:
            raise ValueError("Could not evaluate any cluster count; the data may have too few distinct points")
        k, how = max(scored, key=lambda r: r["silhouette"])["k"], "silhouette"

    # ---- final model -----------------------------------------------------
    km = KMeans(n_clusters=k, n_init=10, random_state=req.random_state).fit(Z)
    labels = km.labels_
    if len(set(labels.tolist())) < k:
        warnings.append("Some clusters ended up empty (too many duplicate points); try a smaller k")
    n_labels = len(set(labels.tolist()))
    metrics = {
        "k": int(k),
        "inertia": _f(km.inertia_),
        "silhouette": _safe_silhouette(Z, labels, req.random_state),
        "davies_bouldin": _f(davies_bouldin_score(Z, labels)) if 2 <= n_labels < n else None,
        "calinski_harabasz": _f(calinski_harabasz_score(Z, labels)) if 2 <= n_labels < n else None,
    }

    feats = [str(c) for c in X.columns]
    centers = scaler.inverse_transform(km.cluster_centers_)
    sizes = np.bincount(labels, minlength=k)
    clusters = [
        {
            "cluster": int(i),
            "size": int(sizes[i]),
            "pct": _f(100 * sizes[i] / n),
            "centroid": {f: _f(v) for f, v in zip(feats, centers[i])},
        }
        for i in range(k)
    ]

    # ---- 2-D projection for plotting ------------------------------------
    if len(feats) == 1:
        pts = np.column_stack([Xi[:, 0], np.zeros(n)])
        cen = np.column_stack([centers[:, 0], np.zeros(k)])
        proj = {"method": "feature", "axes": [feats[0], ""], "explained_variance": None}
    elif len(feats) == 2:
        pts, cen = Xi, centers
        proj = {"method": "features", "axes": feats, "explained_variance": None}
    else:
        pca = PCA(n_components=2, random_state=req.random_state).fit(Z)
        pts, cen = pca.transform(Z), pca.transform(km.cluster_centers_)
        proj = {
            "method": "pca",
            "axes": ["PC1", "PC2"],
            "explained_variance": [_f(v) for v in pca.explained_variance_ratio_],
        }
    proj["points"] = [[_f(a), _f(b)] for a, b in pts]
    proj["centroids"] = [[_f(a), _f(b)] for a, b in cen]

    return {
        "status": "ok",
        "task": "cluster",
        "k": int(k),
        "k_selected_by": how,
        "features": feats,
        "dropped": dropped,
        "rows": {"received": n_received, "used": n},
        "metrics": metrics,
        "k_search": k_search,
        "clusters": clusters,
        "labels": [int(v) for v in labels],
        "row_index": kept_idx,
        "projection": proj,
        "warnings": warnings,
    }


TASKS = {"regression": run_regression, "eda": run_eda, "cluster": run_cluster}


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
def _run_and_callback(req: RegressionRequest) -> None:
    try:
        payload = TASKS[req.task](req)
    except Exception as e:  # noqa: BLE001
        log.exception("%s failed", req.task)
        payload = {"status": "error", "error": str(e)}
    try:
        httpx.post(req.callback_url, json=payload, timeout=30).raise_for_status()
        log.info("Result delivered to callback")
    except Exception as e:  # noqa: BLE001
        log.error("Callback delivery failed: %s", e)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/webhook/regression", dependencies=[Depends(check_key)])
def regression(req: RegressionRequest, background: BackgroundTasks):
    log.info("Request: task=%s, %d rows, model=%s", req.task, len(req.data), req.model_type)
    if req.callback_url:
        background.add_task(_run_and_callback, req)
        return JSONResponse(status_code=202, content={"status": "accepted"})
    try:
        return TASKS[req.task](req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
