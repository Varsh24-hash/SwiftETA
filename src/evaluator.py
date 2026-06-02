"""
evaluator.py  –  Person 3
Comprehensive evaluation of the trained XGBoost ETA model.
Outputs:
  • Console metrics table
  • Models/evaluation_report.json
  • Data/predictions_test.csv  (predicted vs actual)
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ── paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "Data"
MODEL_DIR = "Models"

TARGET_COL   = "ground_truth_eta"
CLASS_TARGET = "is_late"
PROMISED_ETA = 30.0          # SLA threshold (minutes)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_model() -> xgb.XGBRegressor:
    path = os.path.join(MODEL_DIR, "xgboost_eta.pkl")
    return joblib.load(path)


def load_split(split: str) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df  = pd.read_csv(os.path.join(DATA_DIR, f"{split}_features.csv"))
    y_r = df[TARGET_COL]
    y_c = df[CLASS_TARGET]
    X   = df.drop(columns=[TARGET_COL, CLASS_TARGET])
    return X, y_r, y_c


# ── regression metrics ────────────────────────────────────────────────────────

def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    errors  = y_pred - y_true.values
    abs_err = np.abs(errors)
    return {
        "MAE":               round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE":              round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "R2":                round(r2_score(y_true, y_pred), 4),
        "MBE (bias)":        round(float(np.mean(errors)), 4),
        "P90_abs_error":     round(float(np.percentile(abs_err, 90)), 4),
        "P95_abs_error":     round(float(np.percentile(abs_err, 95)), 4),
        "Within_1min_%":     round(float(np.mean(abs_err <= 1.0)) * 100, 2),
        "Within_2.3min_%":   round(float(np.mean(abs_err <= 2.3)) * 100, 2),
        "Within_5min_%":     round(float(np.mean(abs_err <= 5.0)) * 100, 2),
    }


# ── late-delivery classification metrics ─────────────────────────────────────

def late_metrics(y_true: pd.Series,
                 y_pred: np.ndarray,
                 threshold: float = PROMISED_ETA) -> dict:
    pred_late = (y_pred >= threshold).astype(int)
    true_late = y_true.values

    report = classification_report(true_late, pred_late,
                                   target_names=["on_time", "late"],
                                   output_dict=True)
    cm     = confusion_matrix(true_late, pred_late).tolist()
    return {
        "accuracy":         round(accuracy_score(true_late, pred_late), 4),
        "late_precision":   round(report["late"]["precision"], 4),
        "late_recall":      round(report["late"]["recall"], 4),
        "late_f1":          round(report["late"]["f1-score"], 4),
        "confusion_matrix": cm,
        "late_rate_actual": round(float(np.mean(true_late)), 4),
        "late_rate_pred":   round(float(np.mean(pred_late)), 4),
    }


# ── feature importance ────────────────────────────────────────────────────────

def load_feature_importance() -> pd.DataFrame:
    path = os.path.join(MODEL_DIR, "feature_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ── pretty-print helpers ──────────────────────────────────────────────────────

def print_section(title: str, data: dict) -> None:
    width = 40
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    for k, v in data.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for row in v:
                print(f"    {row}")
        else:
            print(f"  {k:28s}: {v}")


def check_target(metrics: dict) -> None:
    mae = metrics.get("MAE", 999)
    ok  = mae <= 2.3
    print(f"\n{'✅' if ok else '❌'}  MAE target ≤ 2.3 min  →  actual MAE = {mae} min")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model …")
    model = load_model()

    report = {}
    all_preds = {}

    for split in ["train", "val", "test"]:
        print(f"\nEvaluating {split} split …")
        X, y_r, y_c = load_split(split)
        preds = model.predict(X)

        reg_m  = regression_metrics(y_r, preds)
        late_m = late_metrics(y_c, preds)

        print_section(f"REGRESSION  [{split.upper()}]", reg_m)
        print_section(f"LATE DELIVERY  [{split.upper()}]", late_m)

        if split == "val":
            check_target(reg_m)

        report[split] = {"regression": reg_m, "late_delivery": late_m}

        # collect predictions
        pred_df = pd.DataFrame({
            "y_true":    y_r.values,
            "y_pred":    preds.round(2),
            "abs_error": np.abs(preds - y_r.values).round(2),
            "is_late_true": y_c.values,
            "is_late_pred": (preds >= PROMISED_ETA).astype(int),
        })
        all_preds[split] = pred_df

    # ── feature importance ─────────────────────────────────────────────────
    fi = load_feature_importance()
    if not fi.empty:
        print(f"\n{'─' * 40}")
        print("  TOP-15 FEATURE IMPORTANCE")
        print(f"{'─' * 40}")
        print(fi.head(15).to_string(index=False))
        report["feature_importance"] = fi.head(15).to_dict(orient="records")

    # ── save artefacts ─────────────────────────────────────────────────────
    report_path = os.path.join(MODEL_DIR, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    pred_path = os.path.join(DATA_DIR, "predictions_test.csv")
    all_preds["test"].to_csv(pred_path, index=False)

    print(f"\nSaved:")
    print(f"  {report_path}")
    print(f"  {pred_path}")