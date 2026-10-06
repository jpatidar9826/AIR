"""Exploratory data analysis for the screening dataset. Saves charts to artifacts/reports/eda/."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from src.data_loader import load_resume_category_data, load_screening_data
from src.features import FeatureBuilder
from src.text_utils import SkillExtractor

OUT = config.EDA_DIR
COLOR = "#1F5F5B"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=120); plt.close(fig)


def main():
    df = load_screening_data()
    summary = {
        "source_file": df.attrs["source_file"], "rows": len(df), "rows_dropped": df.attrs["rows_dropped"],
        "positive_rate": round(float(df.label.mean()), 4), "unique_roles": int(df.job_role.nunique()),
        "resume_words_median": float(df.resume.str.split().str.len().median()),
        "jd_words_median": float(df.job_description.str.split().str.len().median()),
    }

    fig, ax = plt.subplots(figsize=(4, 3.5))
    df.label.map({0: "Not matched", 1: "Matched"}).value_counts().plot.bar(ax=ax, color=COLOR)
    ax.set_title("Label distribution"); ax.set_xlabel(""); save(fig, "label_distribution.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    df.job_role.value_counts().head(15).sort_values().plot.barh(ax=ax, color=COLOR)
    ax.set_title("Top 15 job roles"); save(fig, "top_roles.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    df.resume.str.split().str.len().plot.hist(bins=40, ax=ax, color=COLOR)
    ax.set_title("Resume length (words)"); save(fig, "resume_length.png")

    skills = SkillExtractor.from_corpus(df.resume.tolist())
    top = pd.Series([s for t in df.resume for s in skills.extract(t)]).value_counts().head(20)
    fig, ax = plt.subplots(figsize=(7, 5))
    top.sort_values().plot.barh(ax=ax, color=COLOR); ax.set_title("Top 20 skills in resumes")
    save(fig, "top_skills.png")

    sample = df.sample(min(len(df), 3000), random_state=config.RANDOM_STATE)
    fb = FeatureBuilder(skills).fit(sample.resume, sample.job_description)
    feats = fb.transform(sample)
    feats["label"] = sample.label.values
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    feats.boxplot(column="tfidf_similarity", by="label", ax=axes[0])
    feats.boxplot(column="skill_coverage", by="label", ax=axes[1])
    fig.suptitle("Do matched candidates have higher similarity and skill coverage?")
    save(fig, "signal_vs_label.png")
    summary["feature_label_correlation"] = feats.corr()["label"].drop("label").round(3).sort_values(
        ascending=False).to_dict()

    try:
        cat = load_resume_category_data()
        fig, ax = plt.subplots(figsize=(7, 6))
        cat.category.value_counts().sort_values().plot.barh(ax=ax, color=COLOR)
        ax.set_title("Resume categories"); save(fig, "resume_categories.png")
        summary["category_rows"] = len(cat); summary["categories"] = int(cat.category.nunique())
    except FileNotFoundError:
        pass

    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
