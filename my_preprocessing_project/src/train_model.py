# src/train_model.py

import logging
import os
import pickle

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.utils.config import (
    create_directories,
    get_data_paths,
    get_model_config,
    get_paths,
    get_training_config,
    load_config,
)

# ─── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Data loading ───────────────────────────────────────────────────────────────
def load_dataset(path: str) -> pd.DataFrame:
    """Load a Parquet split into a DataFrame."""
    logger.info(f"Loading data from {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} cols")
    return df


# ─── Model factory ──────────────────────────────────────────────────────────────
def instantiate_model(model_name: str, tuned: bool, cfg: dict):
    """
    Create a model instance based on name and whether to use tuned hyperparameters.
    """
    params = get_model_config(model_name, tuned=tuned, config=cfg)
    logger.info(f"Instantiating {model_name}{' (tuned)' if tuned else ''} with params: {params}")
    if model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "lightgbm":
        return LGBMClassifier(**params)
    elif model_name == "catboost":
        return CatBoostClassifier(**params)
    elif model_name == "randomforest":
        return RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


# ─── Training & evaluation ──────────────────────────────────────────────────────
def prepare_for_training(
    model, X_tr: pd.DataFrame, X_vl: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Encode object cols and clean names, with XGBoost support."""
    X_tr, X_vl = X_tr.copy(), X_vl.copy()
    obj_cols = X_tr.select_dtypes("object").columns

    if isinstance(model, XGBClassifier):
        # encode + rename + enable category...
        # (same logic as before, but all here)
        model.set_params(enable_categorical=True)
    elif obj_cols.any():
        for c in obj_cols:
            X_tr[c] = X_tr[c].astype("category").cat.codes
            X_vl[c] = X_vl[c].astype("category").cat.codes

    return X_tr, X_vl


def compute_metrics(
    y_true, y_pred, y_proba, metrics_list: list[str]
) -> dict[str, float]:
    """Compute and return the requested metrics."""
    results = {}
    for m in metrics_list:
        if m == "accuracy":
            results[m] = accuracy_score(y_true, y_pred)
        elif m == "precision":
            results[m] = precision_score(y_true, y_pred, zero_division=0)
        elif m == "recall":
            results[m] = recall_score(y_true, y_pred, zero_division=0)
        elif m == "f1":
            results[m] = f1_score(y_true, y_pred, zero_division=0)
        elif m == "roc_auc":
            results[m] = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    return results


def train_and_evaluate(
    model, X_train, y_train, X_val, y_val, metrics_list
):
    # 1) prepare data
    X_tr, X_vl = prepare_for_training(model, X_train, X_val)

    # 2) fit & predict
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_vl)
    try:
        y_proba = model.predict_proba(X_vl)[:, 1]
    except Exception:
        y_proba = None

    # 3) compute metrics
    results = compute_metrics(y_val, y_pred, y_proba, metrics_list)
    logger.info(f"Validation metrics: {results}")
    return results



# ─── Model persistence ───────────────────────────────────────────────────────────
def save_model(model, path: str):
    """Save a trained model to disk (using pickle)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(f"Saving model to {path}")
    with open(path, "wb") as f:
        pickle.dump(model, f)


# ─── Main workflow ─────────────────────────────────────────────────────────────
def main():
    # 1) Load all config sections
    cfg        = load_config()
    data_cfg   = get_data_paths(cfg)["processed"]
    train_cfg  = get_training_config(cfg)
    paths_cfg  = get_paths(cfg)

    # 2) Ensure directories exist
    create_directories()

    # 3) Load train/val data
    train_df = load_dataset(data_cfg["train"])
    val_df   = load_dataset(data_cfg["validation"])

    X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    X_val,   y_val   = val_df.drop("target", axis=1),   val_df["target"]

    # 4) Choose model (example: tuned XGBoost)
    model_name = "xgboost"           # or pull from CLI/config
    tuned      = True
    metrics    = train_cfg.get("metrics", [])
    model      = instantiate_model(model_name, tuned, cfg)

    # 5) Train & evaluate
    results = train_and_evaluate(model, X_train, y_train, X_val, y_val, metrics)

    # 6) Save the trained model
    model_filename = f"{model_name}_{'tuned' if tuned else 'baseline'}.pkl"
    save_path = os.path.join(paths_cfg["models"], model_filename)
    save_model(model, save_path)

    # 7) Optionally: save the results dict to disk
    results_fp = os.path.join(paths_cfg["results"], f"{model_name}_metrics.json")
    pd.Series(results).to_json(results_fp)
    logger.info(f"Saved metrics to {results_fp}")


if __name__ == "__main__":
    main()
