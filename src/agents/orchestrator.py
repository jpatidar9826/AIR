"""Orchestrator: runs the agents in sequence and records a trace for the UI.

Resume Analyzer -> JD Matcher (+ ML) -> GenAI summary & questions
-> Panel Recommendation -> RAG policy check
"""
from __future__ import annotations

import time

import config
from src.agents.base import Trace
from src.agents.jd_matcher import JDMatcherAgent
from src.agents.panel_agent import PanelRecommendationAgent
from src.agents.resume_analyzer import ResumeAnalyzerAgent
from src.genai import candidate_summary, interview_questions
from src.llm.client import LLMClient
from src.ml.predict import load_predictors
from src.rag.qa import answer_question
from src.rag.store import PolicyStore
from src.text_utils import SkillExtractor


class RecruitmentOrchestrator:
    def __init__(self):
        self.skills = SkillExtractor.load(config.SKILL_VOCAB_PATH)
        self.shortlist, self.category = load_predictors()
        self.llm = LLMClient()
        self.analyzer = ResumeAnalyzerAgent(self.skills, self.category)
        self.matcher = JDMatcherAgent(self.skills, self.shortlist)
        self.panel = PanelRecommendationAgent()
        try:
            self.policies = PolicyStore.load_or_build()
        except FileNotFoundError:
            self.policies = None

    def status(self) -> dict:
        return {
            "Shortlist model": self.shortlist.name if self.shortlist else None,
            "Category model": self.category.name if self.category else None,
            "Interviewer pool": f"{len(self.panel.panel)} interviewers" if self.panel.ready else None,
            "Policy index": f"{len(self.policies.chunks)} chunks" if self.policies else None,
            "LLM": f"{self.llm.provider} / {self.llm.model}" if self.llm.available else None,
        }

    def screen(self, resume: str, jd: str, role: str = "", n_questions: int = 8,
               use_genai: bool = True) -> dict:
        trace = Trace()

        t = time.time()
        profile = self.analyzer.run(resume)
        trace.record(self.analyzer.name, "Extract skills, education, experience",
                     f"{len(profile['skills'])} skills, {profile['experience_years']:g} yrs, {profile['education']}", t)

        t = time.time()
        match = self.matcher.run(resume, jd, role, profile)
        trace.record(self.matcher.name, "Score candidate vs JD (rules + ML)",
                     f"final {match['final_score']} -> {match['recommendation']['decision']}", t)

        summary, questions = None, None
        if use_genai:
            t = time.time()
            summary = candidate_summary(self.llm, resume, jd, role, profile, match)
            questions = interview_questions(self.llm, jd, role, profile, match, n_questions)
            trace.record("GenAI", "Write summary + interview questions",
                         f"summary ({summary['mode']}), {len(questions['questions'])} questions ({questions['mode']})", t)

        t = time.time()
        panel = self.panel.run(role, match["jd_skills"], profile["skills"], profile["experience_years"])
        trace.record(self.panel.name, "Select interview panel", f"{len(panel)} interviewers", t)

        policy = None
        if self.policies is not None:
            t = time.time()
            policy = answer_question("How should the interview panel be formed and what must interviewers "
                                     "do to ensure a fair, structured interview?", self.policies, self.llm, k=3)
            trace.record("HR Policy RAG", "Retrieve interview-process policy", f"{len(policy['sources'])} sources", t)

        return {"profile": profile, "match": match, "summary": summary, "questions": questions,
                "panel": panel, "policy": policy, "trace": trace.as_rows()}

    def rank(self, df, jd: str, role: str = "") -> "pd.DataFrame":  # noqa: F821
        """Fast batch ranking (no GenAI) for many resumes against one JD."""
        rows = []
        for i, resume in enumerate(df["resume"].astype(str)):
            profile = self.analyzer.run(resume)
            m = self.matcher.run(resume, jd, role, profile)
            rows.append({"row": i, "final_score": m["final_score"], "match_score": m["match_score"],
                         "ml_probability": round(m["ml_probability"], 3) if m["ml_probability"] is not None else None,
                         "decision": m["recommendation"]["decision"],
                         "matched_skills": ", ".join(m["matched_skills"][:6]),
                         "missing_skills": ", ".join(m["missing_skills"][:6]),
                         "experience_years": profile["experience_years"], "education": profile["education"],
                         "resume_preview": resume[:160]})
        import pandas as pd
        return pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
