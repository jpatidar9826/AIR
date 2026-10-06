# AI-Powered Recruitment Platform

Automates candidate screening with **ML**, **RAG**, **GenAI** and **Agents**, delivered through a Streamlit app.

| Layer | What it does | Where |
|---|---|---|
| ML | Shortlist prediction (4 models compared) + resume category classifier (3 models compared) | `src/ml/` |
| RAG | Answers HR policy questions with cited sources | `src/rag/` |
| GenAI | Candidate summaries and tailored interview questions | `src/genai.py` |
| Agents | Resume Analyzer, JD Matcher, Panel Recommendation, coordinated by an orchestrator | `src/agents/` |
| UI | Screening, batch ranking, policy chat, model insights | `app.py` |

## 1. Datasets (public, Kaggle)

| Dataset | Used for |
|---|---|
| [surendra365/recruitement-dataset](https://www.kaggle.com/datasets/surendra365/recruitement-dataset) | Resume + job description + best-match label: shortlist model, JD matcher, batch ranking |
| [gauravduttakiit/resume-dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) | Resume text + category: resume category classifier |

Download the CSVs and place them anywhere under `data/raw/` (sub-folders are fine), or run
`python run_pipeline.py --download` if the environment has internet and Kaggle credentials.
The loader detects the right files by their columns. If it picks the wrong file, set
`SCREENING_FILE` / `RESUME_CATEGORY_FILE` in `.env`.

**Not from Kaggle (state this in the report):**
- `data/hr_policies/*.md`: sample HR policies written for this project, for the RAG demo.
- `data/processed/interviewers_synthetic.csv`: synthetic interviewer pool generated from the dataset's roles and skills (no public interviewer dataset exists).

## 2. Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # keep LLM_PROVIDER=offline until MAAS access is confirmed
```

## 3. Build everything

```bash
python run_pipeline.py      # EDA -> train models -> interviewer pool -> RAG index + evaluation
pytest -q                   # smoke tests
streamlit run app.py
```

No Kaggle data yet? `python scripts/make_demo_data.py` creates a small **fake** dataset in
`data/raw/_demo/` purely to check the code runs. Delete that folder before the real run and
never report its results.

## 4. LLM access (MAAS)

All LLM calls go through `src/llm/client.py`. Configure the endpoint allowed by the MAAS
guidelines (AI Lab VM: Projects -> MAAS) in `.env`:

```
LLM_PROVIDER=openai_compatible   # or azure
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
```

- With `LLM_PROVIDER=offline`, summaries, questions and policy answers use deterministic templates / extractive answers, so the demo always works.
- Emails, phone numbers and URLs are masked before any text is sent (`MASK_PII_BEFORE_LLM=true`).
- Every call is logged (no content) to `artifacts/reports/llm_usage_log.jsonl` as evidence of compliant use.
- All prompts are in `src/llm/prompts.py` and include guardrails against using protected attributes.

## 5. How scoring works

**Features** (`src/features.py`): TF-IDF similarity, skill overlap / coverage / Jaccard, education level and gap,
experience years and gap, certifications, resume length, role-title presence.
Demographic columns (age, gender, race, ethnicity, name) are **never** used as features.

**Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost with 5-fold stratified CV and a held-out
test set (accuracy, precision, recall, F1, ROC-AUC). The best model by F1 is saved.

**Final score** = 0.6 x ML probability x 100 + 0.4 x rule score, where the rule score weights skill coverage (45%),
semantic similarity (30%), experience fit (15%) and education fit (10%).
Shortlist >= 65, Hold 45-65, Reject < 45. Recommendations are decision support only.

**Fairness audit**: selection rate per sensitive group on the test set with the four-fifths rule
(`artifacts/reports/fairness_report.json`).

**RAG evaluation**: Hit@1, Hit@3 and MRR on `data/hr_policies/eval_questions.json`.

## 6. Project structure

```
app.py                  Streamlit UI
run_pipeline.py         one-command build
config.py               paths, datasets, model and LLM settings
src/data_loader.py      Kaggle loading + column auto-detection
src/text_utils.py       cleaning, skill mining/extraction, education/experience parsing, PII masking
src/features.py         resume-JD pair features
src/ml/train.py         model training, comparison, plots, fairness audit
src/ml/predict.py       inference wrappers
src/rag/store.py        document loading, chunking, retrieval
src/rag/qa.py           cited answers (LLM or extractive)
src/llm/                LLM client + prompts
src/genai.py            summaries + interview questions
src/agents/             resume analyzer, JD matcher, panel recommender, orchestrator
scripts/                EDA, panel builder, RAG evaluation, demo data
artifacts/reports/      everything you need for the presentation
docs/SPRINT_PLAN.md     sprint plan and work split
```

## 7. AI Lab reminders

Sessions end 6 hours after launch: commit code, `artifacts/reports/` and notes to the repository
regularly. Don't upload confidential data anywhere.
