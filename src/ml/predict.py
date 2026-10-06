"""Inference wrappers used by the agents and the Streamlit app."""
from __future__ import annotations

import joblib
import numpy as np

import config
from src.text_utils import clean_text


class ShortlistPredictor:
    def __init__(self, path=config.SHORTLIST_MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.name = bundle["model_name"]
        self.fb = bundle["feature_builder"]
        self.threshold = bundle["threshold"]
        self.metrics = bundle["metrics"]
        self.importance = bundle.get("feature_importance", {})

    def predict(self, resume: str, jd: str, role: str = "") -> dict:
        X = self.fb.transform_one(resume, jd, role)
        prob = float(self.model.predict_proba(X)[0, 1])
        return {"probability": prob, "shortlisted": prob >= self.threshold,
                "model": self.name, "features": X.iloc[0].round(3).to_dict()}

    def predict_batch(self, df):
        X = self.fb.transform(df)
        return self.model.predict_proba(X)[:, 1]


class CategoryPredictor:
    def __init__(self, path=config.CATEGORY_MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.name = bundle["model_name"]

    def predict(self, resume: str, top_k: int = 3) -> list[tuple[str, float]]:
        text = [clean_text(resume)]
        clf = self.model
        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(text)[0]
        else:  # LinearSVC -> softmax over decision scores
            d = clf.decision_function(text)[0]
            scores = np.exp(d - d.max()); scores /= scores.sum()
        classes = clf.classes_
        order = np.argsort(scores)[::-1][:top_k]
        return [(str(classes[i]), float(scores[i])) for i in order]


def load_predictors():
    sp = ShortlistPredictor() if config.SHORTLIST_MODEL_PATH.exists() else None
    cp = CategoryPredictor() if config.CATEGORY_MODEL_PATH.exists() else None
    return sp, cp
