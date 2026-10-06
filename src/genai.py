"""GenAI: candidate summaries and interview questions (LLM with offline fallback)."""
from __future__ import annotations

from src.llm.client import LLMClient, parse_json
from src.llm.prompts import QUESTIONS_SYSTEM, QUESTIONS_USER, SUMMARY_SYSTEM, SUMMARY_USER
from src.text_utils import EDU_NAMES


def _trim(text: str, n: int = 3500) -> str:
    return text if len(text) <= n else text[:n] + " ..."


def candidate_summary(llm: LLMClient, resume: str, jd: str, role: str, analysis: dict, match: dict) -> dict:
    prob = match.get("ml_probability")
    prompt = SUMMARY_USER.format(
        role=role or "Not specified", jd=_trim(jd), resume=_trim(resume),
        match_score=match["match_score"], prob=f"{prob:.0%}" if prob is not None else "n/a",
        matched=", ".join(match["matched_skills"]) or "none",
        missing=", ".join(match["missing_skills"]) or "none")
    reply = llm.chat(SUMMARY_SYSTEM, prompt, purpose="candidate_summary")
    if reply:
        return {"text": reply, "mode": "llm"}

    # ---- offline template
    edu = EDU_NAMES.get(analysis["education_level"], "Not stated")
    strengths = match["matched_skills"][:5]
    gaps = match["missing_skills"][:4]
    rec = match["recommendation"]
    lines = [
        "**Profile**",
        f"Candidate with about {analysis['experience_years']:g} years of experience "
        f"(education: {edu}) and {len(analysis['skills'])} identifiable skills"
        + (f", most aligned with {analysis['predicted_category'][0][0]}." if analysis.get("predicted_category") else "."),
        "",
        "**Strengths for this role**",
        *([f"- Demonstrates {s}" for s in strengths] or ["- No direct skill overlap detected"]),
        f"- Semantic similarity to the JD: {match['semantic_similarity']:.0%}",
        "",
        "**Gaps or risks to verify**",
        *([f"- No evidence of {g}" for g in gaps] or ["- No major skill gaps detected"]),
    ]
    if match["experience_gap"] < 0:
        lines.append(f"- Experience is about {abs(match['experience_gap']):g} years below the stated requirement")
    lines += ["", f"**Recommendation:** {rec['decision']} - {rec['reason']}"]
    return {"text": "\n".join(lines), "mode": "template"}


def interview_questions(llm: LLMClient, jd: str, role: str, analysis: dict, match: dict, n: int = 8) -> dict:
    prompt = QUESTIONS_USER.format(
        role=role or "Not specified", jd=_trim(jd, 2500),
        matched=", ".join(match["matched_skills"]) or "none",
        missing=", ".join(match["missing_skills"]) or "none",
        exp=analysis["experience_years"], n=n)
    parsed = parse_json(llm.chat(QUESTIONS_SYSTEM, prompt, purpose="interview_questions"))
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        return {"questions": parsed[:n], "mode": "llm"}

    # ---- offline template bank
    role_name = role or "this role"
    qs = []
    for s in match["matched_skills"][:3]:
        qs.append({"category": "Technical", "question": f"Walk us through a recent piece of work where you used {s}. "
                   "What was your specific contribution and what was the result?",
                   "what_to_look_for": f"Concrete, first-hand detail about {s}; measurable outcome."})
    for s in match["missing_skills"][:2]:
        qs.append({"category": "Gap probing", "question": f"This role needs {s}. What exposure do you have to it, "
                   "and how would you get up to speed in your first month?",
                   "what_to_look_for": "Honest self-assessment, adjacent experience, a realistic learning plan."})
    qs += [
        {"category": "Behavioural", "question": "Tell us about a time you had to deliver under a tight deadline "
         "with incomplete information. What did you do?",
         "what_to_look_for": "Prioritisation, communication with stakeholders, ownership."},
        {"category": "Role scenario", "question": f"In your first 90 days as {role_name}, what would you focus on "
         "and how would you measure success?", "what_to_look_for": "Understanding of the role, structured plan."},
        {"category": "Behavioural", "question": "Describe a disagreement with a teammate and how it was resolved.",
         "what_to_look_for": "Collaboration, respect, focus on outcome over ego."},
        {"category": "Motivation", "question": f"What draws you to {role_name}, and what do you want to learn next?",
         "what_to_look_for": "Genuine alignment with the role and growth mindset."},
    ]
    return {"questions": qs[:n], "mode": "template"}
