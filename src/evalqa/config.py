"""Settings for the simulated rating round and the QA analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIMENSIONS = ["accuracy", "instruction_following", "relevance", "safety", "clarity"]


@dataclass(frozen=True)
class Config:
    seed: int = int(os.getenv("SEED", 11))
    raters_per_item: int = 4
    audit_share: float = 0.25          # share of evaluations sampled for QA audit
    # recalibration thresholds
    max_mae: float = 0.60              # mean absolute error vs gold, across dimensions
    max_abs_bias: float = 0.40         # average signed difference vs gold
    min_detection: float = 0.75        # share of planted defects the rater caught
    min_pref_accuracy: float = 0.85    # share of pairs where the rater picked the better response
    data_dir: Path = ROOT / "data"
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", str(ROOT / "reports"))))
    prompts_dir: Path = ROOT / "prompts"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


CONFIG = Config()
