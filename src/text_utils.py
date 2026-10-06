"""Text cleaning, skill extraction, education/experience parsing and PII masking."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

BASE_SKILLS = [
    # programming & data
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r programming", "scala", "golang", "ruby", "php",
    "html", "css", "react", "angular", "node.js", "django", "flask", "spring", "rest api",
    "machine learning", "deep learning", "nlp", "computer vision", "data analysis", "data science",
    "statistics", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "spark",
    "hadoop", "tableau", "power bi", "excel", "etl", "data visualization", "big data",
    # cloud & devops
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "linux", "ci/cd", "terraform",
    "devops", "microservices", "networking", "cybersecurity", "network security",
    # business
    "project management", "agile", "scrum", "stakeholder management", "business analysis",
    "product management", "digital marketing", "seo", "content writing", "sales", "negotiation",
    "customer service", "accounting", "financial analysis", "budgeting", "auditing", "taxation",
    "recruitment", "talent acquisition", "payroll", "employee relations", "supply chain",
    "logistics", "operations management", "procurement", "crm", "sap", "erp",
    # soft skills
    "communication", "leadership", "teamwork", "problem solving", "time management",
    "critical thinking", "presentation", "mentoring",
    # healthcare / other domains
    "patient care", "pharmacology", "nursing", "medical terminology", "healthcare", "nutrition",
    "teaching", "curriculum development", "graphic design", "ui/ux", "figma", "photoshop",
    "autocad", "mechanical design", "electrical engineering", "civil engineering",
]

EDUCATION_LEVELS = [
    (4, r"\b(ph\.?d|doctorate|doctoral)\b"),
    (3, r"\b(master'?s?|mba|m\.?tech|m\.?sc|m\.?s\.?|m\.?e\.?|mca|post[- ]?graduate)\b"),
    (2, r"\b(bachelor'?s?|b\.?tech|b\.?sc|b\.?e\.?|b\.?com|bca|bba|graduate degree|undergraduate)\b"),
    (1, r"\b(diploma|associate degree)\b"),
    (0, r"\b(high school|secondary school|12th)\b"),
]
EDU_NAMES = {0: "High school", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}

SENIORITY_YEARS = [
    (r"\b(intern|fresher|entry[- ]level|graduate trainee)\b", 0.5),
    (r"\b(junior)\b", 1.5),
    (r"\b(mid[- ]level|intermediate)\b", 4.0),
    (r"\b(senior|sr\.)\b", 7.0),
    (r"\b(lead|principal|staff|head of|manager)\b", 9.0),
]


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mask_pii(text: str) -> str:
    """Remove emails, phone numbers and URLs before text leaves the machine."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", str(text))
    text = re.sub(r"(\+?\d[\d\s().-]{8,}\d)", "[PHONE]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)
    return text


def mine_skills_from_corpus(texts, min_freq: int = 3) -> list[str]:
    """Learn domain skills from phrases like 'Proficient in X, Y, Z' / 'Skills: X, Y'."""
    patterns = [
        r"proficient in ([^.;\n]+?)(?:,? with |\.|;|\n|$)",
        r"skills?\s*[:\-]\s*([^.\n]+)",
        r"skilled in ([^.;\n]+?)(?:,? with |\.|;|\n|$)",
    ]
    counter: Counter = Counter()
    for text in texts:
        low = str(text).lower()
        for pat in patterns:
            for match in re.findall(pat, low):
                for part in re.split(r",| and |\|", match):
                    skill = re.sub(r"[^a-z0-9+#./\s-]", "", part).strip(" .-")
                    if 2 <= len(skill) <= 40 and 1 <= len(skill.split()) <= 4:
                        counter[skill] += 1
    return sorted(s for s, c in counter.items() if c >= min_freq)


class SkillExtractor:
    def __init__(self, vocab=None):
        self.vocab = sorted({s.lower().strip() for s in (vocab or BASE_SKILLS) if s.strip()},
                            key=len, reverse=True)
        escaped = [re.escape(s) for s in self.vocab]
        self._regex = re.compile(r"(?<![a-z0-9])(" + "|".join(escaped) + r")(?![a-z0-9+#])")

    def extract(self, text: str) -> list[str]:
        return sorted(set(self._regex.findall(str(text).lower())))

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.vocab, indent=1))

    @classmethod
    def load(cls, path: Path) -> "SkillExtractor":
        p = Path(path)
        return cls(json.loads(p.read_text())) if p.exists() else cls()

    @classmethod
    def from_corpus(cls, texts, min_freq: int = 3) -> "SkillExtractor":
        return cls(BASE_SKILLS + mine_skills_from_corpus(texts, min_freq))


def education_level(text: str) -> int:
    low = str(text).lower()
    for level, pattern in EDUCATION_LEVELS:
        if re.search(pattern, low):
            return level
    return -1  # unknown


def experience_years(text: str) -> float:
    low = str(text).lower()
    nums = [float(n) for n in re.findall(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)", low)]
    nums = [n for n in nums if n <= 45]
    if nums:
        return max(nums)
    for pattern, years in reversed(SENIORITY_YEARS):
        if re.search(pattern, low):
            return years
    return 0.0


def certification_count(text: str) -> int:
    low = str(text).lower()
    explicit = re.findall(r"certifications? such as ([^.]+)", low)
    if explicit:
        return sum(len(re.split(r",| and ", e)) for e in explicit)
    return len(re.findall(r"\b(certified|certification|certificate)\b", low))
