"""
model_trainer.py  –  Person 3
Train XGBoost (GBDT) on the engineered feature set.
• Cross-validation on train split to tune hyperparameters.
• Target: MAE ±2.3 min on val set.
• Saves trained model + feature-importance CSV.
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "Data"
MODEL_DIR = "Models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "ground_truth_eta"
CLASS_TARGET = "is_late"

# ── hyperparameter grid (manual grid search via CV) ───────────────────────────
PARAM_GRID = [
    {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
     "subsample": 0.8,  "colsample_bytree": 0.8, "min_child_weight": 3},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.03,
     "subsample": 0.85, "colsample_bytree": 0.75, "min_child_weight": 5},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.05,
     "subsample": 0.9,  "colsample_bytree": 0.9,  "min_child_weight": 3},
]

N_FOLDS = 5


# ── helpers ───────────────────────────────────────────────────────────────────

def load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(DATA_DIR, f"{split}_features.csv")
    df   = pd.read_csv(path)
    drop = [TARGET_COL, CLASS_TARGET]
    X    = df.drop(columns=[c for c in drop if c in df.columns])
    y    = df[TARGET_COL]
    return X, y


def cross_validate(params: dict, X: pd.DataFrame, y: pd.Series) -> float:
    kf   = KFold(n_splits=N_FOLDS, shuffle=False)   # no shuffle – time order
    maes = []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        model = xgb.XGBRegressor(
            **params,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
            verbose=False,
        )
        preds = model.predict(X.iloc[va_idx])
        mae   = mean_absolute_error(y.iloc[va_idx], preds)
        maes.append(mae)
    return float(np.mean(maes))


def train_final(params: dict,
                X_train: pd.DataFrame, y_train: pd.Series,
                X_val:   pd.DataFrame, y_val:   pd.Series) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        **params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=30,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate(model: xgb.XGBRegressor,
             X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    preds  = model.predict(X)
    mae    = mean_absolute_error(y, preds)
    rmse   = np.sqrt(mean_squared_error(y, preds))
    r2     = r2_score(y, preds)
    within = np.mean(np.abs(preds - y) <= 2.3) * 100
    metrics = {
        "split":         split_name,
        "mae_min":       round(mae, 4),
        "rmse_min":      round(rmse, 4),
        "r2":            round(r2, 4),
        "within_2.3min_pct": round(within, 2),
    }
    print(f"\n── {split_name} Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v}")
    return metrics


def save_feature_importance(model: xgb.XGBRegressor,
                            feature_names: list[str]) -> None:
    scores = model.feature_importances_
    fi = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    path = os.path.join(MODEL_DIR, "feature_importance.csv")
    fi.to_csv(path, index=False)
    print(f"\nTop-10 features:")
    print(fi.head(10).to_string(index=False))
    return fi


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading feature splits …")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    print(f"Train : {X_train.shape}   Val : {X_val.shape}   Test : {X_test.shape}")

    # ── cross-validation to pick best hyperparams ──────────────────────────
    print(f"\n{'─'*50}")
    print(f"Running {N_FOLDS}-fold CV across {len(PARAM_GRID)} param sets …")
    best_mae    = float("inf")
    best_params = None

    for i, params in enumerate(PARAM_GRID, 1):
        cv_mae = cross_validate(params, X_train, y_train)
        print(f"  Config {i}: CV-MAE = {cv_mae:.4f} min   params={params}")
        if cv_mae < best_mae:
            best_mae    = cv_mae
            best_params = params

    print(f"\nBest CV-MAE : {best_mae:.4f} min")
    print(f"Best params : {best_params}")

    # ── train final model on full train split ──────────────────────────────
    print(f"\n{'─'*50}")
    print("Training final model …")
    model = train_final(best_params, X_train, y_train, X_val, y_val)

    # ── evaluate ───────────────────────────────────────────────────────────
    all_metrics = []
    all_metrics.append(evaluate(model, X_train, y_train, "train"))
    all_metrics.append(evaluate(model, X_val,   y_val,   "val"))
    all_metrics.append(evaluate(model, X_test,  y_test,  "test"))

    target_met = all_metrics[1]["mae_min"] <= 2.3
    print(f"\n{'✅' if target_met else '❌'}  Val MAE "
          f"{'≤' if target_met else '>'} 2.3 min target "
          f"(actual: {all_metrics[1]['mae_min']} min)")

    # ── save artefacts ─────────────────────────────────────────────────────
    model_path   = os.path.join(MODEL_DIR, "xgboost_eta.json")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    model.save_model(model_path)
    joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_eta.pkl"))

    with open(metrics_path, "w") as f:
        json.dump({
            "best_cv_mae": round(best_mae, 4),
            "best_params": best_params,
            "splits": all_metrics,
        }, f, indent=2)

    save_feature_importance(model, list(X_train.columns))

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  Models/xgboost_eta.pkl")
    print(f"  {metrics_path}")
    print(f"  Models/feature_importance.csv")