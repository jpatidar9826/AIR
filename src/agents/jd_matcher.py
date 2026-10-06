"""JD Matcher Agent: scores a candidate against a job description.

Final score blends a transparent rule-based match score with the ML
shortlist probability (when the model is trained).
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from src.text_utils import SkillExtractor, clean_text, education_level, experience_years

WEIGHTS = {"skills": 0.45, "semantic": 0.30, "education": 0.10, "experience": 0.15}


class JDMatcherAgent:
    name = "JD Matcher"

    def __init__(self, skills: SkillExtractor, shortlist_predictor=None):
        self.skills = skills
        self.predictor = shortlist_predictor

    def _similarity(self, resume: str, jd: str) -> float:
        if self.predictor is not None:
            return float(self.predictor.fb.similarity([resume], [jd])[0])
        vec = TfidfVectorizer(stop_words="english").fit([clean_text(resume), clean_text(jd)])
        m = vec.transform([clean_text(resume), clean_text(jd)])
        return float((m[0] @ m[1].T).toarray()[0, 0])

    def run(self, resume: str, jd: str, role: str, profile: dict) -> dict:
        jd_skills = set(self.skills.extract(jd))
        cand_skills = set(profile["skills"])
        matched = sorted(jd_skills & cand_skills)
        missing = sorted(jd_skills - cand_skills)
        coverage = len(matched) / len(jd_skills) if jd_skills else 0.0
        sim = self._similarity(resume, jd)

        req_edu = education_level(jd)
        edu_fit = 1.0 if req_edu < 0 or profile["education_level"] >= req_edu else 0.4
        req_exp = experience_years(jd)
        exp_gap = profile["experience_years"] - req_exp
        exp_fit = 1.0 if req_exp == 0 or exp_gap >= 0 else max(0.0, 1 + exp_gap / max(req_exp, 1))

        # similarity of short texts rarely exceeds ~0.5, so rescale before weighting
        sem_scaled = min(sim / 0.5, 1.0)
        rule_score = 100 * (WEIGHTS["skills"] * coverage + WEIGHTS["semantic"] * sem_scaled +
                            WEIGHTS["education"] * edu_fit + WEIGHTS["experience"] * exp_fit)

        ml = self.predictor.predict(resume, jd, role) if self.predictor else None
        prob = ml["probability"] if ml else None
        final = 0.6 * prob * 100 + 0.4 * rule_score if prob is not None else rule_score

        return {
            "matched_skills": matched, "missing_skills": missing, "jd_skills": sorted(jd_skills),
            "skill_coverage": coverage, "semantic_similarity": sim,
            "required_experience": req_exp, "experience_gap": exp_gap,
            "education_fit": edu_fit, "match_score": round(rule_score, 1),
            "ml_probability": prob, "ml_model": ml["model"] if ml else None,
            "final_score": round(final, 1),
            "recommendation": self._recommend(final, missing),
            "score_breakdown": {
                "Skill coverage": round(100 * WEIGHTS["skills"] * coverage, 1),
                "Semantic similarity": round(100 * WEIGHTS["semantic"] * sem_scaled, 1),
                "Education fit": round(100 * WEIGHTS["education"] * edu_fit, 1),
                "Experience fit": round(100 * WEIGHTS["experience"] * exp_fit, 1),
            },
        }

    @staticmethod
    def _recommend(score: float, missing: list[str]) -> dict:
        if score >= 65:
            return {"decision": "Shortlist", "reason": "strong overall fit with the job requirements"}
        if score >= 45:
            reason = f"partial fit; verify {', '.join(missing[:2])}" if missing else "partial fit; verify depth of experience"
            return {"decision": "Hold", "reason": reason}
        return {"decision": "Reject", "reason": "low alignment with the core requirements"}
