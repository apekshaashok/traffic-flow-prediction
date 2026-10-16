"""Train and evaluate traffic volume models; writes outputs/results.json and a figure.

Usage:
    python -m src.train                  # downloads the UCI dataset automatically
    python -m src.train --csv path.csv   # use a local copy of the CSV
    python -m src.train --data synthetic # pipeline smoke test only
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_features, load_raw

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
SEED = 42


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def make_models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=SEED),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["uci", "synthetic"], default="uci")
    ap.add_argument("--csv", default=None, help="path to a local Metro_Interstate_Traffic_Volume.csv[.gz]")
    ap.add_argument("--test-frac", type=float, default=0.2)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    raw = load_raw(args.data, args.csv)
    X, y, groups = build_features(raw)

    # Chronological split: never train on the future.
    split = int(len(X) * (1 - args.test_frac))
    X_tr, X_te, y_tr, y_te = X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

    results, preds = [], {}
    feature_sets = {
        "Calendar + weather": groups["calendar_weather"],
        "Calendar + weather + lags": groups["calendar_weather"] + groups["lags"],
    }

    # Baselines.
    hist = y_tr.groupby([X_tr["dow"], X_tr["hour"]]).mean()
    p_hist = np.array([hist.loc[(d, h)] for d, h in zip(X_te["dow"], X_te["hour"])])
    results.append({"features": "Baseline", "model": "Historical mean (day-of-week x hour)", **metrics(y_te, p_hist)})
    p_pers = X_te["lag_1"].values
    results.append({"features": "Baseline", "model": "Persistence (previous hour)", **metrics(y_te, p_pers)})

    for fs_name, cols in feature_sets.items():
        for name, model in make_models().items():
            model.fit(X_tr[cols], y_tr)
            pred = model.predict(X_te[cols])
            preds[(fs_name, name)] = pred
            results.append({"features": fs_name, "model": name, **metrics(y_te, pred)})
            print(f"{fs_name:28s} {name:18s} {results[-1]['MAE']:8.1f} {results[-1]['RMSE']:8.1f} {results[-1]['R2']:6.3f}")

    # ---- Task 2: five-level congestion classification (the CS229 paper's task) ----
    # Level = hourly volume as a share of the maximum training volume, in 20% bands:
    # 0 = free flow ... 4 = severe congestion.
    edges = np.array([0.2, 0.4, 0.6, 0.8]) * y_tr.max()

    def to_level(v):
        return np.digitize(v, edges)

    c_tr, c_te = to_level(y_tr.values), to_level(y_te.values)

    def cls_metrics(y_true, y_pred):
        return {
            "Accuracy": float(accuracy_score(y_true, y_pred)),
            "Within-1": float(np.mean(np.abs(y_true - y_pred) <= 1)),
            "Macro-F1": float(f1_score(y_true, y_pred, average="macro")),
        }

    cls_results = [
        {"features": "Baseline", "model": "Majority level",
         **cls_metrics(c_te, np.full_like(c_te, np.bincount(c_tr).argmax()))},
        {"features": "Baseline", "model": "Historical mean -> level", **cls_metrics(c_te, to_level(p_hist))},
        {"features": "Baseline", "model": "Persistence -> level", **cls_metrics(c_te, to_level(p_pers))},
    ]

    def make_classifiers():
        return {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
            "Gradient Boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=SEED),
            "Neural Net (MLP, 32 hidden)": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, early_stopping=True, random_state=SEED)),
        }

    for fs_name, cols in feature_sets.items():
        for name, model in make_classifiers().items():
            model.fit(X_tr[cols], c_tr)
            cls_results.append({"features": fs_name, "model": name, **cls_metrics(c_te, model.predict(X_te[cols]))})
            print(f"[level] {fs_name:26s} {name:28s} acc {cls_results[-1]['Accuracy']:.3f}")

    # Paper-style reference: one tiny hidden layer (4 neurons) on day + hour only.
    paper = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, early_stopping=True, random_state=SEED))
    paper.fit(X_tr[["dow", "hour"]], c_tr)
    cls_results.append({"features": "Day + hour only", "model": "Paper-style MLP (4 hidden)",
                        **cls_metrics(c_te, paper.predict(X_te[["dow", "hour"]]))})
    print(f"[level] paper-style MLP acc {cls_results[-1]['Accuracy']:.3f}")
    level_share = (np.bincount(c_te, minlength=5) / len(c_te)).round(3).tolist()

    # Best model per feature set for the figure.
    fig, ax = plt.subplots(figsize=(9, 2.8))
    week = slice(0, 24 * 7)
    ax.plot(y_te.index[week], y_te.values[week], color="black", lw=1.4, label="Actual")
    colors = {"Calendar + weather": "tab:orange", "Calendar + weather + lags": "tab:blue"}
    for fs_name in feature_sets:
        best = min((r for r in results if r["features"] == fs_name), key=lambda r: r["RMSE"])["model"]
        ax.plot(y_te.index[week], preds[(fs_name, best)][week], lw=1.1, color=colors[fs_name], label=f"{best} ({fs_name})")
    ax.set_ylabel("Vehicles / hour")
    ax.set_title("First test week: actual vs predicted hourly traffic volume")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pred_vs_actual.png", dpi=170)

    summary = {
        "source": args.data if not args.csv else "uci (local csv)",
        "n_rows": int(len(X)),
        "date_start": str(X.index.min()),
        "date_end": str(X.index.max()),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "n_features_no_lags": len(groups["calendar_weather"]),
        "n_features_with_lags": len(groups["calendar_weather"]) + len(groups["lags"]),
        "target_mean": float(y.mean()),
        "target_max": float(y.max()),
        "results": results,
        "level_edges": [float(e) for e in edges],
        "level_share_test": level_share,
        "classification": cls_results,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {OUT_DIR / 'results.json'} and {OUT_DIR / 'pred_vs_actual.png'}")


if __name__ == "__main__":
    main()
