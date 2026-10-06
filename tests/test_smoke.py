"""Smoke tests: run `python scripts/make_demo_data.py && python run_pipeline.py` first
(or use the real Kaggle data), then `pytest -q`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.store import chunk_document
from src.text_utils import SkillExtractor, education_level, experience_years, mask_pii


def test_text_utils():
    t = "Proficient in Python, SQL and Machine Learning with 5 years experience. Masters degree. a@b.com"
    assert {"python", "sql", "machine learning"} <= set(SkillExtractor().extract(t))
    assert education_level(t) == 3
    assert experience_years(t) == 5
    assert "[EMAIL]" in mask_pii(t)


def test_chunking():
    chunks = chunk_document("doc.md", "# Title\n\n" + "\n\n".join(["Sentence number one here."] * 80))
    assert len(chunks) > 1 and all(len(c.text) < 1000 for c in chunks)


def test_orchestrator_end_to_end():
    from src.agents.orchestrator import RecruitmentOrchestrator
    orc = RecruitmentOrchestrator()
    out = orc.screen("Proficient in Python, SQL, machine learning, with mid-level experience. Masters degree.",
                     "Hiring a Data Scientist. Required skills: python, machine learning, statistics. 2+ years.",
                     "Data Scientist")
    assert 0 <= out["match"]["final_score"] <= 100
    assert out["questions"]["questions"]
    assert "python" in out["match"]["matched_skills"]
