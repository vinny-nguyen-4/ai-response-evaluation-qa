"""Simulated rating round.

The test set (``data/eval_items.json``) has 32 prompts, each with a good response and a
flawed one containing a single planted defect. Because the defect is known, every
response has a **gold** score: what a careful reviewer following the v2 guidelines
would give. Eight reviewers with different habits then rate the pairs under the v1
guidelines, so the QA analysis has realistic mistakes to find.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG, DIMENSIONS, Config

# Which dimensions each planted defect should lower, and to what (gold scores).
DEFECT_GOLD = {
    "factual_error": {"accuracy": 1},
    "hallucinated_source": {"accuracy": 2},
    "missing_requirement": {"instruction_following": 2},
    "unsafe": {"safety": 1, "accuracy": 2},
    "padding": {"relevance": 3},
}

ISSUE_TEXT = {
    "factual_error": ("accuracy", "states an incorrect fact or calculation"),
    "hallucinated_source": ("accuracy", "cites a study or source that can't be verified"),
    "missing_requirement": ("instruction following", "leaves out part of what the prompt asked for"),
    "unsafe": ("safety", "gives advice that could cause harm"),
    "padding": ("relevance", "adds background the user didn't ask for"),
}


@dataclass(frozen=True)
class Rater:
    id: str
    profile: str
    research: float      # chance of catching factual errors (and, scaled down, made-up sources)
    safety: float        # chance of catching unsafe advice
    instructions: float  # chance of catching a missed requirement
    bias: float          # added to every score
    noise: float         # sd of random scoring noise
    padding_dim: str     # where this reviewer puts the penalty for off-topic padding under v1
    mismatch: float      # chance the rationale names a problem the scores don't reflect


RATERS = [
    Rater("R01", "Careful", 0.97, 0.98, 0.95, 0.0, 0.25, "relevance", 0.02),
    Rater("R02", "Steady", 0.88, 0.95, 0.92, 0.0, 0.35, "relevance", 0.05),
    Rater("R03", "Skips research", 0.45, 0.92, 0.90, 0.0, 0.35, "clarity", 0.05),
    Rater("R04", "Lenient", 0.85, 0.92, 0.88, 0.55, 0.35, "clarity", 0.05),
    Rater("R05", "Harsh", 0.88, 0.95, 0.92, -0.5, 0.35, "relevance", 0.05),
    Rater("R06", "Safety gaps", 0.85, 0.55, 0.90, 0.0, 0.35, "clarity", 0.05),
    Rater("R07", "Inconsistent", 0.78, 0.88, 0.85, 0.0, 0.85, "relevance", 0.08),
    Rater("R08", "New hire", 0.62, 0.85, 0.70, 0.1, 0.55, "clarity", 0.25),
]


def load_items(cfg: Config = CONFIG) -> pd.DataFrame:
    return pd.DataFrame(json.loads((cfg.data_dir / "eval_items.json").read_text()))


def gold_scores(defect: str | None) -> dict[str, int]:
    scores = {d: 5 for d in DIMENSIONS}
    if defect:
        scores.update(DEFECT_GOLD[defect])
    return scores


def _detect_prob(r: Rater, defect: str) -> float:
    return {"factual_error": r.research, "hallucinated_source": r.research * 0.75,
            "missing_requirement": r.instructions, "unsafe": r.safety, "padding": 0.95}[defect]


def simulate_round(items: pd.DataFrame, cfg: Config = CONFIG) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (responses, ratings, truth). ``truth`` holds what the simulator knows
    (whether each rater actually noticed the defect) and is used only for testing."""
    rng = np.random.default_rng(cfg.seed)
    n_r = len(RATERS)

    # randomize which side (A/B) holds the flawed response
    resp_rows = []
    for _, it in items.iterrows():
        flawed_side = "A" if rng.random() < 0.5 else "B"
        for side in "AB":
            flawed = side == flawed_side
            resp_rows.append({"item_id": it["id"], "side": side, "flawed": flawed,
                              "defect": it["defect"] if flawed else None,
                              "text": it["flawed"] if flawed else it["good"],
                              **{f"gold_{d}": v for d, v in gold_scores(it["defect"] if flawed else None).items()}})
    responses = pd.DataFrame(resp_rows)

    ratings, truth = [], []
    for i, it in items.reset_index(drop=True).iterrows():
        assigned = [RATERS[(i + 2 * k) % n_r] for k in range(cfg.raters_per_item)]
        resp = responses[responses["item_id"] == it["id"]].set_index("side")
        flawed_side = resp.index[resp["flawed"]][0]
        good_side = "B" if flawed_side == "A" else "A"
        for r in assigned:
            noticed = rng.random() < _detect_prob(r, it["defect"])
            row = {"eval_id": f"{it['id']}-{r.id}", "item_id": it["id"], "rater_id": r.id}
            issues = []
            for side in "AB":
                is_flawed = side == flawed_side
                base = {d: 5 for d in DIMENSIONS}
                if is_flawed and noticed:
                    if it["defect"] == "padding":
                        base[r.padding_dim] = 3
                    else:
                        base.update(DEFECT_GOLD[it["defect"]])
                    issues.append(side)
                for d in DIMENSIONS:
                    s = base[d] + r.bias + rng.normal(0, r.noise)
                    row[f"{side}_{d}"] = int(np.clip(round(s), 1, 5))
                # rationale names the problem but the score wasn't lowered (QA should catch this)
                if is_flawed and noticed and rng.random() < r.mismatch:
                    key = "relevance" if it["defect"] == "padding" else list(DEFECT_GOLD[it["defect"]])[0]
                    row[f"{side}_{key}"] = max(row[f"{side}_{key}"], 4)
            if noticed:
                pref = good_side if rng.random() < 0.97 else flawed_side
            else:
                pref = good_side if rng.random() < 0.5 else flawed_side
            row["preferred"] = pref
            if noticed:
                dim, txt = ISSUE_TEXT[it["defect"]]
                if it["defect"] == "padding" and r.padding_dim == "clarity":
                    dim = "clarity"
                row["rationale"] = f"{flawed_side} {txt} ({dim}). {good_side} answers the prompt correctly."
            else:
                row["rationale"] = "Both responses answer the prompt; no significant issues found."
            ratings.append(row)
            truth.append({"eval_id": row["eval_id"], "noticed": noticed, "defect": it["defect"],
                          "flawed_side": flawed_side, "good_side": good_side})
    return responses, pd.DataFrame(ratings), pd.DataFrame(truth)
