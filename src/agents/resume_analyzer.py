"""Resume Analyzer Agent: turns raw resume text into a structured profile."""
from __future__ import annotations

import re

from src.text_utils import (EDU_NAMES, SkillExtractor, certification_count,
                            education_level, experience_years)

SECTION_HEADERS = ["summary", "objective", "experience", "work history", "education", "skills",
                   "projects", "certifications", "achievements"]


class ResumeAnalyzerAgent:
    name = "Resume Analyzer"

    def __init__(self, skills: SkillExtractor, category_predictor=None):
        self.skills = skills
        self.category_predictor = category_predictor

    def run(self, resume: str) -> dict:
        low = resume.lower()
        edu = education_level(resume)
        profile = {
            "skills": self.skills.extract(resume),
            "education_level": edu,
            "education": EDU_NAMES.get(edu, "Not stated"),
            "experience_years": experience_years(resume),
            "certifications": certification_count(resume),
            "sections_found": [h for h in SECTION_HEADERS if re.search(rf"\b{h}\b", low)],
            "word_count": len(resume.split()),
            "has_contact_info": bool(re.search(r"\S+@\S+|\+?\d[\d\s-]{8,}", resume)),
        }
        if self.category_predictor is not None:
            profile["predicted_category"] = self.category_predictor.predict(resume)
        # simple quality flags a recruiter would care about
        flags = []
        if profile["word_count"] < 80:
            flags.append("Very short resume - limited evidence")
        if not profile["skills"]:
            flags.append("No recognisable skills found")
        if edu < 0:
            flags.append("Education not stated")
        profile["quality_flags"] = flags
        return profile
