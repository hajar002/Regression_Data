"""
Regression analysis server for n8n.

n8n (Google Sheets / Supabase node) -> HTTP Request node -> POST /webhook/regression
-> JSON response with the fitted regression model (metrics, coefficients, predictions).

Optionally pass `callback_url` (an n8n Webhook node URL) to run asynchronously:
the server replies 202 immediately and POSTs the result to that URL when done.
"""
import hmac
import logging
import math
import os
from typing import Any, Literal, Optional

import httpx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("regression-server")

API_KEY = (os.getenv("API_KEY") or "").strip() or None  # optional; if set, clients must send X-API-Key
MIN_ROWS = 5

app = FastAPI(title="n8n Regression Server", version="1.0.0")


# --------------------------------------------------------------------------- #
# Auth
# --------------------------------------------------------------------------- #
def check_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header (no header with that name was received)")
    received = x_api_key.strip()  # tolerate stray spaces / line breaks
    if not hmac.compare_digest(received.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail=f"Invalid X-API-Key (received {len(received)} characters)")


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
# Routes
# --------------------------------------------------------------------------- #
def _run_and_callback(req: RegressionRequest) -> None:
    try:
        payload = run_regression(req)
    except Exception as e:  # noqa: BLE001
        log.exception("Regression failed")
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
    log.info("Regression request: %d rows, model=%s", len(req.data), req.model_type)
    if req.callback_url:
        background.add_task(_run_and_callback, req)
        return JSONResponse(status_code=202, content={"status": "accepted"})
    try:
        return run_regression(req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
