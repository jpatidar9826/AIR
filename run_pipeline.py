"""One command to build everything the app needs.

    python run_pipeline.py                 # eda + train + panel + rag
    python run_pipeline.py --download      # also fetch datasets from Kaggle first
    python run_pipeline.py --steps train rag
"""
import argparse
import time

STEPS = ["eda", "train", "panel", "rag"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="download Kaggle datasets with kagglehub")
    ap.add_argument("--steps", nargs="+", default=STEPS, choices=STEPS)
    args = ap.parse_args()

    if args.download:
        from src.data_loader import download_from_kaggle
        download_from_kaggle()

    for step in args.steps:
        t = time.time()
        print(f"\n===== {step.upper()} =====")
        if step == "eda":
            from scripts.eda import main as eda
            eda()
        elif step == "train":
            from src.ml.train import train_category_model, train_shortlist_model
            train_shortlist_model()
            train_category_model()
        elif step == "panel":
            import config
            from src.agents.panel_agent import build_synthetic_panel
            from src.data_loader import load_screening_data
            from src.text_utils import SkillExtractor
            panel = build_synthetic_panel(load_screening_data(), SkillExtractor.load(config.SKILL_VOCAB_PATH))
            print(f"{len(panel)} synthetic interviewers saved")
        elif step == "rag":
            from scripts.evaluate_rag import main as rag
            rag()
        print(f"({step} took {time.time() - t:.1f}s)")
    print("\nDone. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
