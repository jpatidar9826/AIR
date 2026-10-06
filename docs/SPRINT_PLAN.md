# Sprint plan and work allocation

Adjust names and the number of members. Each member owns one area end to end **and**
pairs on one other, so everyone can explain their work and the overall solution.

| Member | Primary ownership | Secondary | Key files | Evidence for the review |
|---|---|---|---|---|
| Member 1 | Problem analysis, datasets, EDA | Documentation | `data_loader.py`, `scripts/eda.py` | Dataset justification, EDA charts, data quality notes |
| Member 2 | Feature engineering + ML models | Fairness audit | `features.py`, `ml/train.py` | Model comparison table, ROC, feature importance, fairness report |
| Member 3 | RAG + HR policy corpus | Prompt engineering | `rag/`, `data/hr_policies/` | Hit@k / MRR, example cited answers |
| Member 4 | GenAI + LLM integration (MAAS) | Testing | `llm/`, `genai.py` | Prompts, LLM usage log, before/after examples |
| Member 5 | Agents + Streamlit app | Demo + presentation | `agents/`, `app.py` | Agent trace, live demo script |

## Sprints

| Sprint | Goal | Done when |
|---|---|---|
| 1. Understand and source data | Requirements, Kaggle datasets, EDA | `artifacts/reports/eda/` committed, dataset section of report written |
| 2. ML | Features, 4 models, CV, best model saved | `model_comparison.csv` + plots + fairness report |
| 3. RAG | Policy corpus, chunking, retrieval, evaluation | Hit@3 >= 0.8 on eval questions |
| 4. GenAI | Summaries and interview questions via MAAS-approved LLM | Prompt file reviewed, outputs reviewed for bias |
| 5. Agents + UI | Orchestrator, panel agent, Streamlit pages | End-to-end demo runs on a fresh session |
| 6. Test and present | Tests, documentation, presentation, demo rehearsal | `pytest` passes, slides cover all 7 required sections |

## Sprint log template (one per sprint)

- Sprint / dates:
- Goal:
- Work done (by member):
- Results / metrics:
- Challenges and how we handled them:
- Next steps:
