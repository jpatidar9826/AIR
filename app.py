"""Streamlit front end for the AI-Powered Recruitment Platform.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json

import pandas as pd
import streamlit as st

import config
from src.agents.orchestrator import RecruitmentOrchestrator
from src.rag.qa import answer_question

_V = tuple(int(x) for x in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _V >= (1, 46) else {"use_container_width": True}

st.set_page_config(page_title="Recruitment Assistant", page_icon="🧭", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.decision {display:inline-block; padding:.25rem .8rem; border-radius:6px; font-weight:600;}
.Shortlist {background:#DCEFE9; color:#1F5F5B;}
.Hold {background:#FFF1D6; color:#8A5A00;}
.Reject {background:#F8E1E1; color:#8F2D2D;}
.skill {display:inline-block; margin:2px 4px 2px 0; padding:2px 8px; border-radius:4px;
        font-size:.85rem; background:#EEF3F2;}
.skill.miss {background:#F8E1E1;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached --
@st.cache_resource(show_spinner="Loading models and indexes...")
def get_orchestrator() -> RecruitmentOrchestrator:
    return RecruitmentOrchestrator()


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame | None:
    try:
        from src.data_loader import load_screening_data
        return load_screening_data()
    except FileNotFoundError:
        return None


def read_upload(file) -> str:
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)
    if name.endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", errors="ignore")


def chips(items, missing=False):
    if not items:
        return "<span style='color:#6b7b7c'>None</span>"
    cls = "skill miss" if missing else "skill"
    return "".join(f"<span class='{cls}'>{s}</span>" for s in items)


orc = get_orchestrator()
data = get_dataset()

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("Recruitment Assistant")
    page = st.radio("Go to", ["Overview", "Screen a candidate", "Rank candidates",
                              "HR policy assistant", "Model insights"], label_visibility="collapsed")
    st.divider()
    st.caption("System status")
    for k, v in orc.status().items():
        st.markdown(f"{'✅' if v else '⚪'} **{k}**: {v or 'not available'}")
    if not orc.llm.available:
        st.caption("LLM is offline, so summaries and questions use templates. "
                   "Set the MAAS-approved endpoint in `.env` to enable it.")


# ---------------------------------------------------------------- overview --
if page == "Overview":
    st.title("AI-powered recruitment platform")
    st.write("Screen resumes against a job description, predict shortlisting, generate interview kits, "
             "recommend interview panels and answer HR policy questions.")

    if orc.shortlist is None:
        st.warning("No trained model found. Put the Kaggle CSVs in `data/raw/` and run "
                   "`python run_pipeline.py`, then reload this page.")

    c1, c2, c3, c4 = st.columns(4)
    if data is not None:
        c1.metric("Applications", f"{len(data):,}")
        c2.metric("Job roles", data.job_role.nunique())
        c3.metric("Match rate", f"{data.label.mean():.0%}")
    if orc.shortlist:
        c4.metric(f"Best model F1 ({orc.shortlist.name})", f"{orc.shortlist.metrics['f1']:.3f}")

    st.subheader("How a candidate is processed")
    st.markdown("""
1. **Resume Analyzer agent** extracts skills, education, experience and certifications, and predicts the resume category (ML).
2. **JD Matcher agent** compares the resume with the job description and blends a transparent rule score with the ML shortlist probability.
3. **GenAI** writes a candidate summary and a tailored set of interview questions.
4. **Panel Recommendation agent** picks interviewers who cover the role's skills and have availability.
5. **HR policy RAG** pulls the relevant interview-process policy so the panel follows it.
""")
    st.subheader("Datasets")
    for key, meta in config.KAGGLE_DATASETS.items():
        st.markdown(f"- [{meta['slug']}]({meta['url']}): {meta['purpose']}")
    st.caption("The interviewer pool is synthetic, generated from roles and skills in the screening data. "
               "Sample HR policies were written for this project.")

    eda = config.EDA_DIR
    imgs = [p for p in ["label_distribution.png", "top_roles.png", "top_skills.png", "signal_vs_label.png"]
            if (eda / p).exists()]
    if imgs:
        st.subheader("Exploratory data analysis")
        cols = st.columns(2)
        for i, img in enumerate(imgs):
            cols[i % 2].image(str(eda / img), **WIDE)


# ------------------------------------------------------- screen a candidate --
elif page == "Screen a candidate":
    st.title("Screen a candidate")

    if data is not None:
        with st.expander("Load a sample from the dataset"):
            roles = sorted(data.job_role.unique())
            pick_role = st.selectbox("Job role", roles)
            if st.button("Load random sample"):
                row = data[data.job_role == pick_role].sample(1).iloc[0]
                st.session_state.update(resume_text=row.resume, jd_text=row.job_description,
                                        role_text=row.job_role, sample_label=int(row.label))

    left, right = st.columns(2)
    with left:
        up = st.file_uploader("Upload resume (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"])
        if up is not None:
            st.session_state["resume_text"] = read_upload(up)
        resume = st.text_area("Resume text", key="resume_text", height=260)
    with right:
        role = st.text_input("Job role", key="role_text")
        jd = st.text_area("Job description", key="jd_text", height=260)

    o1, o2 = st.columns([1, 3])
    n_q = o1.slider("Interview questions", 4, 12, 8)
    use_genai = o2.toggle("Generate summary and interview questions", value=True)

    if st.button("Screen candidate", type="primary", disabled=not (resume and jd)):
        with st.spinner("Running agents..."):
            st.session_state["result"] = orc.screen(resume, jd, role, n_q, use_genai)

    res = st.session_state.get("result")
    if res:
        m, p = res["match"], res["profile"]
        rec = m["recommendation"]
        st.divider()
        a, b, c, d = st.columns(4)
        a.metric("Final score", f"{m['final_score']}/100")
        b.metric("Rule match score", f"{m['match_score']}/100")
        b.caption("Skills, similarity, education, experience")
        c.metric("ML shortlist probability", f"{m['ml_probability']:.0%}" if m["ml_probability"] is not None else "n/a")
        if m["ml_model"]:
            c.caption(m["ml_model"])
        d.markdown(f"<div style='margin-top:.6rem'>Recommendation</div>"
                   f"<span class='decision {rec['decision']}'>{rec['decision']}</span>"
                   f"<div style='font-size:.85rem;margin-top:.3rem'>{rec['reason']}</div>",
                   unsafe_allow_html=True)
        if "sample_label" in st.session_state:
            st.caption(f"Dataset label for this sample: {'Match' if st.session_state['sample_label'] else 'No match'}")

        t1, t2, t3, t4, t5 = st.tabs(["Match details", "Summary", "Interview kit", "Panel", "Agent trace"])
        with t1:
            st.markdown("**Matched skills**<br>" + chips(m["matched_skills"]), unsafe_allow_html=True)
            st.markdown("**Missing skills**<br>" + chips(m["missing_skills"], True), unsafe_allow_html=True)
            st.bar_chart(pd.Series(m["score_breakdown"], name="points"), horizontal=True, color="#1F5F5B")
            pc = st.columns(4)
            pc[0].metric("Experience", f"{p['experience_years']:g} yrs",
                         f"{m['experience_gap']:+g} vs required" if m["required_experience"] else None)
            pc[1].metric("Education", p["education"])
            pc[2].metric("Certifications", p["certifications"])
            pc[3].metric("Similarity", f"{m['semantic_similarity']:.0%}")
            if p.get("predicted_category"):
                st.write("**Predicted resume category:** " +
                         ", ".join(f"{c} ({s:.0%})" for c, s in p["predicted_category"]))
            for f in p["quality_flags"]:
                st.warning(f)
        with t2:
            if res["summary"]:
                st.markdown(res["summary"]["text"])
                st.caption(f"Generated by: {res['summary']['mode']}")
            else:
                st.info("Turn on summary generation to see this.")
        with t3:
            if res["questions"]:
                for i, q in enumerate(res["questions"]["questions"], 1):
                    st.markdown(f"**{i}. {q.get('question')}**  \n"
                                f"*{q.get('category', '')}*: look for {q.get('what_to_look_for', '')}")
                st.caption(f"Generated by: {res['questions']['mode']}")
        with t4:
            if res["panel"]:
                st.dataframe(pd.DataFrame(res["panel"]), hide_index=True, **WIDE)
                st.caption("Interviewer pool is synthetic demo data.")
            else:
                st.info("No interviewer pool found. Run `python run_pipeline.py --steps panel`.")
            if res["policy"]:
                st.markdown("**What the interview policy says**")
                st.write(res["policy"]["answer"])
                st.caption("Sources: " + ", ".join(sorted({s['doc'] for s in res['policy']['sources']})))
        with t5:
            st.dataframe(pd.DataFrame(res["trace"]), hide_index=True, **WIDE)

        report = {"role": role, "match": m, "profile": p,
                  "summary": res["summary"]["text"] if res["summary"] else None,
                  "questions": res["questions"]["questions"] if res["questions"] else None,
                  "panel": res["panel"]}
        st.download_button("Download screening report (JSON)", json.dumps(report, indent=2, default=str),
                           file_name="screening_report.json")
        st.caption("This is decision support. A recruiter reviews every recommendation before any candidate is rejected.")


# ---------------------------------------------------------- rank candidates --
elif page == "Rank candidates":
    st.title("Rank candidates for a job")
    source = st.radio("Candidates", ["From dataset", "Upload CSV"], horizontal=True)

    pool, role, jd = None, "", ""
    if source == "From dataset":
        if data is None:
            st.error("Dataset not found in data/raw/.")
        else:
            role = st.selectbox("Job role", sorted(data.job_role.unique()))
            subset = data[data.job_role == role]
            jd = st.text_area("Job description", value=subset.job_description.iloc[0], height=160)
            n = st.slider("Number of applicants to rank", 10, min(300, max(10, len(subset))), min(50, len(subset)))
            pool = subset.sample(min(n, len(subset)), random_state=1).reset_index(drop=True)
    else:
        f = st.file_uploader("CSV with a 'resume' column", type="csv")
        role = st.text_input("Job role")
        jd = st.text_area("Job description", height=160)
        if f is not None:
            pool = pd.read_csv(f)
            pool.columns = [c.lower().strip() for c in pool.columns]
            if "resume" not in pool.columns:
                st.error("The CSV needs a column named 'resume'."); pool = None

    if pool is not None and jd and st.button("Rank candidates", type="primary"):
        with st.spinner(f"Scoring {len(pool)} resumes..."):
            ranked = orc.rank(pool, jd, role)
        if "label" in pool.columns:
            ranked["dataset_label"] = ranked["row"].map(pool["label"])
        st.dataframe(ranked, hide_index=True, **WIDE)
        counts = ranked.decision.value_counts()
        st.write(" | ".join(f"**{k}**: {v}" for k, v in counts.items()))
        if "dataset_label" in ranked:
            top = ranked.head(max(1, len(ranked) // 5))
            st.caption(f"Precision in top 20%: {top.dataset_label.mean():.0%} "
                       f"(base rate {ranked.dataset_label.mean():.0%})")
        st.download_button("Download ranking (CSV)", ranked.to_csv(index=False), "ranking.csv")


# ------------------------------------------------------ HR policy assistant --
elif page == "HR policy assistant":
    st.title("HR policy assistant")
    if orc.policies is None:
        st.error("No policy documents found in data/hr_policies/.")
    else:
        st.caption("Answers come only from: " + ", ".join(orc.policies.documents))
        st.session_state.setdefault("chat", [])
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        q = st.chat_input("Ask about hiring, interviews, offers, data privacy...")
        if q:
            st.session_state.chat.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            out = answer_question(q, orc.policies, orc.llm)
            with st.chat_message("assistant"):
                st.markdown(out["answer"])
                if out["sources"]:
                    with st.expander(f"Sources ({out['mode']})"):
                        for s in out["sources"]:
                            st.markdown(f"**[{s['n']}] {s['doc']}: {s['section']}** (score {s['score']})")
                            st.caption(s["text"])
            st.session_state.chat.append({"role": "assistant", "content": out["answer"]})


# ----------------------------------------------------------- model insights --
elif page == "Model insights":
    st.title("Model insights")
    rep = config.REPORTS_DIR
    if (rep / "model_comparison.csv").exists():
        st.subheader("Shortlist prediction: model comparison")
        st.dataframe(pd.read_csv(rep / "model_comparison.csv"), hide_index=True, **WIDE)
        cols = st.columns(3)
        for col, img in zip(cols, ["roc_curves.png", "confusion_matrix.png", "feature_importance.png"]):
            if (rep / img).exists():
                col.image(str(rep / img), **WIDE)
    else:
        st.info("Train the models first: `python run_pipeline.py --steps train`.")

    if (rep / "fairness_report.json").exists():
        st.subheader("Fairness audit")
        st.caption("Sensitive attributes are excluded from the features. This checks the selection rate "
                   "per group on the test set (four-fifths rule: ratio should be at least 0.8).")
        fair = json.loads((rep / "fairness_report.json").read_text())
        if fair:
            for attr, r in fair.items():
                st.markdown(f"**{attr.title()}**: disparate impact ratio {r['disparate_impact_ratio']} "
                            f"{'(passes)' if r['passes_four_fifths_rule'] else '(needs review)'}")
                st.bar_chart(pd.Series(r["selection_rate"], name="selection rate"), color="#1F5F5B")
        else:
            st.write("No sensitive columns in the dataset to audit.")

    if (rep / "category_model_comparison.csv").exists():
        st.subheader("Resume category classifier")
        st.dataframe(pd.read_csv(rep / "category_model_comparison.csv"), hide_index=True)

    if (rep / "rag_evaluation.json").exists():
        st.subheader("RAG retrieval evaluation")
        r = json.loads((rep / "rag_evaluation.json").read_text())
        c = st.columns(3)
        c[0].metric("Hit@1", r["hit@1"]); c[1].metric("Hit@3", r["hit@3"]); c[2].metric("MRR", r["mrr"])

    if config.LLM_LOG_PATH.exists():
        st.subheader("LLM usage log")
        log = pd.read_json(config.LLM_LOG_PATH, lines=True)
        st.dataframe(log.tail(50), hide_index=True, **WIDE)
