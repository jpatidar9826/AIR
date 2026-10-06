"""Central configuration: paths, dataset sources, model and LLM settings."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POLICY_DIR = DATA_DIR / "hr_policies"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
EDA_DIR = REPORTS_DIR / "eda"

for _d in (RAW_DIR, PROCESSED_DIR, POLICY_DIR, MODELS_DIR, REPORTS_DIR, EDA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- datasets --
# Public Kaggle datasets used by the project (document these in the report).
KAGGLE_DATASETS = {
    "screening": {
        "slug": "surendra365/recruitement-dataset",
        "url": "https://www.kaggle.com/datasets/surendra365/recruitement-dataset",
        "purpose": "Resume + job description + best-match label -> shortlist model, JD matcher",
    },
    "resume_category": {
        "slug": "gauravduttakiit/resume-dataset",
        "url": "https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset",
        "purpose": "Resume text + category -> resume category classifier (Resume Analyzer)",
    },
}

# Columns that must never be used as model features (kept only for the fairness audit).
SENSITIVE_COLUMNS = ["age", "gender", "race", "ethnicity", "job_applicant_name", "name"]

# ------------------------------------------------------------------- model --
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SHORTLIST_THRESHOLD = float(os.getenv("SHORTLIST_THRESHOLD", "0.5"))

SHORTLIST_MODEL_PATH = MODELS_DIR / "shortlist_model.joblib"
CATEGORY_MODEL_PATH = MODELS_DIR / "category_model.joblib"
SKILL_VOCAB_PATH = MODELS_DIR / "skills_vocab.json"
RAG_INDEX_PATH = MODELS_DIR / "rag_index.joblib"
PANEL_PATH = PROCESSED_DIR / "interviewers_synthetic.csv"
LLM_LOG_PATH = REPORTS_DIR / "llm_usage_log.jsonl"

# --------------------------------------------------------------------- LLM --
# Fill these from the MAAS guidelines inside the AI Lab VM (Projects -> MAAS).
# LLM_PROVIDER: "openai_compatible" | "azure" | "offline"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "offline").lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")
LLM_API_VERSION = os.getenv("LLM_API_VERSION", "2024-02-01")  # azure only
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
MASK_PII_BEFORE_LLM = os.getenv("MASK_PII_BEFORE_LLM", "true").lower() == "true"

# --------------------------------------------------------------------- RAG --
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "false").lower() == "true"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RAG_CHUNK_SIZE = 700
RAG_CHUNK_OVERLAP = 120
RAG_TOP_K = 4
RAG_MIN_SCORE = 0.05
