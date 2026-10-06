"""Panel Recommendation Agent.

No public dataset of company interviewers exists, so we generate a clearly
labelled SYNTHETIC interviewer pool from the job roles and skills in the
Kaggle screening data (scripts/build_panel.py). Replace it with real HR data
in production.
"""
from __future__ import annotations

import random
from collections import Counter

import pandas as pd

import config

FIRST = ["Asha", "Rahul", "Meera", "Vikram", "Priya", "Arjun", "Kavya", "Rohan", "Neha", "Sanjay",
         "Divya", "Karthik", "Ananya", "Aditya", "Ishita", "Manoj", "Pooja", "Suresh", "Lakshmi", "Nikhil"]
LAST = ["Iyer", "Sharma", "Nair", "Reddy", "Gupta", "Menon", "Patel", "Rao", "Das", "Kulkarni"]
LEVELS = {"Engineer": 1, "Senior": 2, "Lead": 3, "Manager": 4, "Director": 5}


def build_synthetic_panel(df: pd.DataFrame, skills, per_role: int = 4, max_roles: int = 40,
                          seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, used_names = [], set()
    for role in df.job_role.value_counts().index[:max_roles]:
        texts = df.loc[df.job_role == role, "resume"].head(300)
        counts = Counter(s for t in texts for s in skills.extract(t))
        pool = [s for s, _ in counts.most_common(15)] or ["communication", "problem solving"]
        for _ in range(per_role):
            while True:
                name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                if name not in used_names:
                    used_names.add(name); break
            level = rng.choice(list(LEVELS))
            rows.append({
                "interviewer_id": f"INT{len(rows) + 1:04d}",
                "name": name,
                "primary_role": role,
                "level": level,
                "skills": "; ".join(rng.sample(pool, k=min(len(pool), rng.randint(4, 7)))),
                "interviews_this_week": rng.randint(0, 6),
                "max_per_week": rng.choice([4, 5, 6]),
                "data_source": "SYNTHETIC",
            })
    panel = pd.DataFrame(rows)
    panel.to_csv(config.PANEL_PATH, index=False)
    return panel


class PanelRecommendationAgent:
    name = "Panel Recommendation"

    def __init__(self, panel: pd.DataFrame | None = None):
        if panel is None and config.PANEL_PATH.exists():
            panel = pd.read_csv(config.PANEL_PATH)
        self.panel = panel

    @property
    def ready(self) -> bool:
        return self.panel is not None and not self.panel.empty

    def run(self, role: str, jd_skills: list[str], candidate_skills: list[str],
            candidate_years: float, size: int = 3) -> list[dict]:
        if not self.ready:
            return []
        target = set(jd_skills) | set(candidate_skills)
        cand_level = 1 if candidate_years < 3 else 2 if candidate_years < 6 else 3
        scored = []
        for r in self.panel.itertuples():
            iv_skills = {s.strip() for s in str(r.skills).split(";") if s.strip()}
            overlap = iv_skills & target
            load = r.interviews_this_week / max(r.max_per_week, 1)
            if load >= 1:
                continue  # fully booked
            role_match = 1.0 if role and role.lower() == str(r.primary_role).lower() else 0.0
            senior_ok = 1.0 if LEVELS.get(r.level, 1) >= cand_level else 0.3
            skill_fit = min(1.0, 3 * len(overlap) / max(len(target), 1))
            score = 0.45 * skill_fit + 0.30 * role_match + 0.15 * senior_ok + 0.10 * (1 - load)
            scored.append((score, r, overlap, role_match, load))
        scored.sort(key=lambda x: x[0], reverse=True)

        # greedy: prefer interviewers who add new skill coverage, mix of levels
        chosen, covered, levels = [], set(), set()
        for score, r, overlap, role_match, load in scored:
            gain = len(overlap - covered)
            if chosen and gain == 0 and r.level in levels and len(scored) > size * 2:
                continue
            chosen.append({
                "name": r.name, "level": r.level, "primary_role": r.primary_role,
                "score": round(score, 3), "covers_skills": sorted(overlap),
                "workload": f"{r.interviews_this_week}/{r.max_per_week} this week",
                "reason": self._reason(overlap, role_match, r.level, load),
                "data_source": r.data_source,
            })
            covered |= overlap; levels.add(r.level)
            if len(chosen) == size:
                break
        return chosen

    @staticmethod
    def _reason(overlap, role_match, level, load) -> str:
        parts = []
        if role_match:
            parts.append("works in the same role")
        if overlap:
            parts.append(f"can assess {', '.join(sorted(overlap)[:3])}")
        parts.append(f"{level.lower()} level")
        if load < 0.5:
            parts.append("has availability")
        return "; ".join(parts).capitalize()
