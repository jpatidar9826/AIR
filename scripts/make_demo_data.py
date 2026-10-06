"""Create a SMALL FAKE dataset ONLY to smoke-test the code when Kaggle data is not
yet available. Never use these results in the report or presentation.

Writes data/raw/_demo/*.csv - delete that folder once the real Kaggle CSVs are in data/raw/.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

ROLES = {
    "Data Scientist": ["python", "machine learning", "statistics", "sql", "pandas", "deep learning", "nlp"],
    "Web Developer": ["javascript", "react", "html", "css", "node.js", "rest api", "git"],
    "HR Manager": ["recruitment", "employee relations", "payroll", "communication", "talent acquisition"],
    "Accountant": ["accounting", "taxation", "auditing", "excel", "financial analysis", "budgeting"],
    "DevOps Engineer": ["aws", "docker", "kubernetes", "linux", "ci/cd", "terraform", "jenkins"],
}
DEGREES = ["Bachelors", "Masters", "PhD", "Diploma"]
LEVELS = ["entry-level", "mid-level", "senior-level"]


def main(n=1500, seed=7):
    rng = random.Random(seed)
    rows = []
    all_skills = sorted({s for v in ROLES.values() for s in v})
    for i in range(n):
        role = rng.choice(list(ROLES))
        match = rng.random() < 0.5
        own = rng.sample(ROLES[role], k=rng.randint(3, 5)) if match else rng.sample(all_skills, k=4)
        resume = (f"Proficient in {', '.join(own)}, with {rng.choice(LEVELS)} experience in the field. "
                  f"Holds a {rng.choice(DEGREES)} degree. Skilled in delivering results.")
        jd = (f"We are hiring a {role}. Required skills: {', '.join(ROLES[role][:5])}. "
              f"Bachelor's degree and 2+ years of experience preferred.")
        label = int(match) if rng.random() > 0.1 else int(not match)  # 10% noise
        rows.append({"Job Applicant Name": f"Person {i}", "Age": rng.randint(21, 55),
                     "Gender": rng.choice(["Male", "Female"]), "Resume": resume,
                     "Job Roles": role, "Job Description": jd, "Best Match": label})
    out = config.RAW_DIR / "_demo"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "demo_screening.csv", index=False)
    cats = [{"Category": r, "Resume": "Experienced professional skilled in " + ", ".join(rng.sample(s, 3)) * 3}
            for r, s in ROLES.items() for _ in range(40)]
    pd.DataFrame(cats).to_csv(out / "demo_resume_category.csv", index=False)
    print(f"Wrote FAKE demo data to {out}")


if __name__ == "__main__":
    main()
