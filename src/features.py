"""Feature engineering for (resume, job description) pairs.

Only job-relevant signals are used. Demographic columns are never features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.text_utils import (SkillExtractor, certification_count, clean_text,
                            education_level, experience_years)

FEATURE_NAMES = [
    "tfidf_similarity",      # semantic overlap of resume and JD
    "skill_overlap",         # number of JD skills found in resume
    "skill_coverage",        # share of JD skills found in resume
    "skill_jaccard",         # overlap / union of skills
    "resume_skill_count",
    "jd_skill_count",
    "education_level",
    "required_education",
    "education_gap",         # candidate level - required level
    "experience_years",
    "required_experience",
    "experience_gap",
    "certifications",
    "resume_word_count",
    "role_title_in_resume",  # job role words present in resume
]


class FeatureBuilder:
    def __init__(self, skill_extractor: SkillExtractor | None = None):
        self.skills = skill_extractor or SkillExtractor()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                          sublinear_tf=True, stop_words="english")
        self.fitted = False

    def fit(self, resumes, jds) -> "FeatureBuilder":
        corpus = [clean_text(t) for t in list(resumes) + list(jds)]
        self.vectorizer.fit(corpus)
        self.fitted = True
        return self

    def similarity(self, resumes, jds) -> np.ndarray:
        r = self.vectorizer.transform([clean_text(t) for t in resumes])
        j = self.vectorizer.transform([clean_text(t) for t in jds])
        return np.asarray(r.multiply(j).sum(axis=1)).ravel()  # rows are L2-normalised

    def _pair(self, resume: str, jd: str, role: str = "") -> dict:
        rs, js = set(self.skills.extract(resume)), set(self.skills.extract(jd))
        overlap = len(rs & js)
        union = len(rs | js) or 1
        edu, req_edu = education_level(resume), education_level(jd)
        exp, req_exp = experience_years(resume), experience_years(jd)
        role_words = [w for w in clean_text(role).split() if len(w) > 2]
        res_low = clean_text(resume)
        role_hit = np.mean([w in res_low for w in role_words]) if role_words else 0.0
        return {
            "skill_overlap": overlap,
            "skill_coverage": overlap / len(js) if js else 0.0,
            "skill_jaccard": overlap / union,
            "resume_skill_count": len(rs),
            "jd_skill_count": len(js),
            "education_level": edu,
            "required_education": req_edu,
            "education_gap": (edu - req_edu) if edu >= 0 and req_edu >= 0 else 0,
            "experience_years": exp,
            "required_experience": req_exp,
            "experience_gap": exp - req_exp,
            "certifications": certification_count(resume),
            "resume_word_count": len(str(resume).split()),
            "role_title_in_resume": float(role_hit),
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("FeatureBuilder must be fitted first")
        roles = df["job_role"] if "job_role" in df else [""] * len(df)
        rows = [self._pair(r, j, role) for r, j, role in zip(df["resume"], df["job_description"], roles)]
        feats = pd.DataFrame(rows, index=df.index)
        feats["tfidf_similarity"] = self.similarity(df["resume"], df["job_description"])
        return feats[FEATURE_NAMES].astype(float)

    def transform_one(self, resume: str, jd: str, role: str = "") -> pd.DataFrame:
        return self.transform(pd.DataFrame({"resume": [resume], "job_description": [jd], "job_role": [role]}))
