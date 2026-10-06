"""Generate the SYNTHETIC interviewer pool used by the Panel Recommendation Agent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.agents.panel_agent import build_synthetic_panel
from src.data_loader import load_screening_data
from src.text_utils import SkillExtractor

if __name__ == "__main__":
    panel = build_synthetic_panel(load_screening_data(), SkillExtractor.load(config.SKILL_VOCAB_PATH))
    print(f"Saved {len(panel)} synthetic interviewers to {config.PANEL_PATH.relative_to(config.ROOT)}")
