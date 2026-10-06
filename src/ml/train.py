"""Train and compare candidate-shortlist models (+ resume category classifier).

Run:  python -m src.ml.train
Outputs (artifacts/):
  models/shortlist_model.joblib     best model + feature builder
  models/category_model.joblib      best resume-category model (if dataset present)
  reports/model_comparison.csv      all models, all metrics
  reports/*.png                     ROC curves, confusion matrix, feature importance
  reports/fairness_report.json      selection rates across sensitive groups
"""
from __future__ import annotations

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import config
from src.data_loader import load_resume_category_data, load_screening_data
from src.features import FEATURE_NAMES, FeatureBuilder
from src.text_utils import SkillExtractor, clean_text


def get_models() -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced",
            random_state=config.RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=config.RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", random_state=config.RANDOM_STATE)
    except ImportError:
        print("xgboost not installed - training 3 models")
    return models


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan"),
    }


def feature_importance(model, names) -> pd.Series:
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    elif hasattr(est, "coef_"):
        vals = np.abs(est.coef_).ravel()
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=names).sort_values(ascending=False)


def fairness_audit(test_df: pd.DataFrame, y_pred) -> dict:
    """Selection rate per group + disparate-impact ratio (4/5ths rule)."""
    report = {}
    for col in [c for c in test_df.columns if c.startswith("sensitive_")]:
        name = col.replace("sensitive_", "")
        if name in ("job_applicant_name", "name"):
            continue
        groups = test_df[col].copy()
        if name == "age" and pd.api.types.is_numeric_dtype(groups):
            groups = pd.cut(groups, bins=[0, 25, 35, 45, 120], labels=["<25", "25-34", "35-44", "45+"])
        rates = pd.Series(y_pred, index=test_df.index).groupby(groups.astype(str)).agg(["mean", "size"])
        rates = rates[rates["size"] >= 20]  # ignore tiny groups
        if len(rates) < 2:
            continue
        di = rates["mean"].min() / rates["mean"].max() if rates["mean"].max() > 0 else float("nan")
        report[name] = {
            "selection_rate": rates["mean"].round(3).to_dict(),
            "group_size": rates["size"].to_dict(),
            "disparate_impact_ratio": round(float(di), 3),
            "passes_four_fifths_rule": bool(di >= 0.8),
        }
    return report


def train_shortlist_model() -> pd.DataFrame:
    t0 = time.time()
    df = load_screening_data()
    print(f"Loaded {len(df)} rows from {df.attrs['source_file']} "
          f"(dropped {df.attrs['rows_dropped']}). Positive rate: {df.label.mean():.2%}")

    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, stratify=df.label,
                                         random_state=config.RANDOM_STATE)

    skills = SkillExtractor.from_corpus(train_df.resume.tolist() + train_df.job_description.tolist())
    skills.save(config.SKILL_VOCAB_PATH)
    print(f"Skill vocabulary: {len(skills.vocab)} terms")

    fb = FeatureBuilder(skills).fit(train_df.resume, train_df.job_description)  # fit on train only
    X_train, X_test = fb.transform(train_df), fb.transform(test_df)
    y_train, y_test = train_df.label.values, test_df.label.values
    print(f"Features built in {time.time() - t0:.1f}s")

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results, fitted, probs = [], {}, {}
    for name, model in get_models().items():
        s = time.time()
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= config.SHORTLIST_THRESHOLD).astype(int)
        m = evaluate(y_test, pred, prob)
        m.update({"model": name, "cv_f1_mean": cv_f1.mean(), "cv_f1_std": cv_f1.std(),
                  "train_seconds": time.time() - s})
        results.append(m)
        fitted[name], probs[name] = model, prob
        print(f"  {name:20s} F1={m['f1']:.3f}  AUC={m['roc_auc']:.3f}  CV-F1={cv_f1.mean():.3f}±{cv_f1.std():.3f}")

    comp = pd.DataFrame(results).set_index("model").sort_values(["f1", "roc_auc"], ascending=False)
    comp.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv")
    best_name = comp.index[0]
    best = fitted[best_name]
    best_pred = (probs[best_name] >= config.SHORTLIST_THRESHOLD).astype(int)
    print(f"Best model: {best_name}")

    # ---- plots
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={comp.loc[name, 'roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "roc_curves.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, best_pred, display_labels=["Not shortlisted", "Shortlisted"],
                                            ax=ax, colorbar=False, cmap="Greens")
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=120); plt.close(fig)

    fi = feature_importance(best, FEATURE_NAMES)
    if not fi.empty:
        fi.round(4).to_csv(config.REPORTS_DIR / "feature_importance.csv", header=["importance"])
        fig, ax = plt.subplots(figsize=(6, 5))
        fi.sort_values().plot.barh(ax=ax, color="#1F5F5B")
        ax.set_title(f"Feature importance - {best_name}")
        fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=120); plt.close(fig)

    fairness = fairness_audit(test_df, best_pred)
    (config.REPORTS_DIR / "fairness_report.json").write_text(json.dumps(fairness, indent=2))
    if fairness:
        for attr, r in fairness.items():
            print(f"  fairness[{attr}] disparate impact = {r['disparate_impact_ratio']} "
                  f"({'OK' if r['passes_four_fifths_rule'] else 'REVIEW'})")

    joblib.dump({
        "model": best, "model_name": best_name, "feature_builder": fb, "feature_names": FEATURE_NAMES,
        "metrics": comp.loc[best_name].to_dict(), "threshold": config.SHORTLIST_THRESHOLD,
        "feature_importance": fi.to_dict(), "trained_rows": len(train_df),
        "source_file": df.attrs["source_file"],
    }, config.SHORTLIST_MODEL_PATH)
    print(f"Saved {config.SHORTLIST_MODEL_PATH.relative_to(config.ROOT)} in {time.time() - t0:.1f}s")
    return comp


def train_category_model() -> pd.DataFrame | None:
    try:
        df = load_resume_category_data()
    except FileNotFoundError as exc:
        print(f"Skipping category model: {exc}")
        return None
    counts = df.category.value_counts()
    df = df[df.category.isin(counts[counts >= 5].index)]
    X = df.resume.map(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        X, df.category, test_size=config.TEST_SIZE, stratify=df.category, random_state=config.RANDOM_STATE)

    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000,
                               sublinear_tf=True, stop_words="english")

    models = {
        "Multinomial NB": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=3000, C=5))]),
        "Linear SVM": Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(C=1.0))]),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, pred),
                     "macro_f1": f1_score(y_test, pred, average="macro"),
                     "weighted_f1": f1_score(y_test, pred, average="weighted")})
        fitted[name] = m
        print(f"  {name:20s} acc={rows[-1]['accuracy']:.3f} macroF1={rows[-1]['macro_f1']:.3f}")
    comp = pd.DataFrame(rows).set_index("model").sort_values("macro_f1", ascending=False)
    comp.round(4).to_csv(config.REPORTS_DIR / "category_model_comparison.csv")
    best = comp.index[0]
    (config.REPORTS_DIR / "category_classification_report.txt").write_text(
        classification_report(y_test, fitted[best].predict(X_test), zero_division=0))
    joblib.dump({"model": fitted[best], "model_name": best, "metrics": comp.loc[best].to_dict(),
                 "classes": sorted(df.category.unique())}, config.CATEGORY_MODEL_PATH)
    print(f"Best category model: {best}")
    return comp


if __name__ == "__main__":
    print("== Shortlist prediction ==")
    train_shortlist_model()
    print("\n== Resume category classification ==")
    train_category_model()
