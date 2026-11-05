import numpy as np
import pandas as pd
import pytest

from evalqa import analysis, judge
from evalqa.config import DIMENSIONS
from evalqa.simulate import RATERS, gold_scores


# ---------------------------------------------------------------- test set
def test_test_set_is_well_formed(round1):
    items, responses, *_ = round1
    assert items["id"].is_unique and len(items) == 32
    assert set(items["defect"]) == {"factual_error", "hallucinated_source", "missing_requirement", "unsafe", "padding"}
    assert (items["good"] != items["flawed"]).all()
    # exactly one flawed response per item, randomly on A or B
    assert (responses.groupby("item_id")["flawed"].sum() == 1).all()
    assert 0 < (responses[responses["flawed"]]["side"] == "A").mean() < 1


def test_gold_scores():
    assert gold_scores(None) == {d: 5 for d in DIMENSIONS}
    assert gold_scores("unsafe")["safety"] == 1
    assert gold_scores("padding")["relevance"] == 3 and gold_scores("padding")["clarity"] == 5


def test_ratings_shape_and_range(round1):
    items, _, ratings, _, long = round1
    assert len(ratings) == len(items) * 4
    assert ratings.groupby("item_id")["rater_id"].nunique().eq(4).all()
    assert long[DIMENSIONS].isin([1, 2, 3, 4, 5]).all().all()


# ---------------------------------------------------------------- agreement
def test_alpha_perfect_and_random():
    assert analysis.krippendorff_alpha_interval([[1, 1], [3, 3], [5, 5]]) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = [list(rng.integers(1, 6, 4)) for _ in range(400)]
    assert abs(analysis.krippendorff_alpha_interval(noise)) < 0.1


def test_alpha_known_value():
    # two raters, interval data; worked example: alpha = 1 - D_o / D_e
    units = [[1, 2], [2, 2], [3, 4], [4, 4], [5, 4]]
    vals = np.array([v for u in units for v in u], float)
    n = len(vals)
    d_o = sum((u[0] - u[1]) ** 2 * 2 for u in units) / n
    d_e = ((vals[:, None] - vals[None, :]) ** 2).sum() / (n * (n - 1))
    assert analysis.krippendorff_alpha_interval(units) == pytest.approx(1 - d_o / d_e)


def test_padding_ambiguity_shows_up_as_low_agreement(round1):
    *_, long = round1
    agr = analysis.agreement(long).set_index("dimension")["krippendorff_alpha"]
    assert agr["relevance"] < agr["instruction_following"]
    placement = analysis.penalty_placement(long).set_index("defect")
    assert 0.2 < placement.loc["padding", "share_on_expected"] < 0.8   # reviewers split
    assert placement.loc["factual_error", "share_on_expected"] > 0.9  # no split


# ---------------------------------------------------------------- calibration
def test_detection_matches_simulator_truth(round1):
    _, responses, ratings, truth, long = round1
    flawed = long[long["flawed"]].copy()
    flawed["caught"] = flawed.apply(analysis.caught, axis=1)
    merged = flawed.merge(truth[["eval_id", "noticed"]], on="eval_id")
    # the observable 'caught' rule should agree with what the simulator knows most of the time
    assert (merged["caught"] == merged["noticed"]).mean() > 0.85


def test_scorecards_flag_the_weak_reviewers(round1, cfg):
    _, responses, ratings, _, long = round1
    sc = analysis.scorecards(long, ratings, responses, cfg).set_index("rater_id")
    assert sc.loc["R01", "status"] == "On target"
    assert sc.loc["R05", "bias"] < -cfg.max_abs_bias            # harsh reviewer
    assert sc.loc["R06", "status"] == "Recalibrate"               # safety gaps
    assert len(sc) == len(RATERS)


# ---------------------------------------------------------------- QA audit
def test_audit_checks(round1):
    _, responses, ratings, _, long = round1
    f = analysis.audit_findings(ratings, long, responses)
    assert set(f["check"]) <= {"missed_defect", "rationale_mismatch", "wrong_preference", "misapplied_dimension", "false_flag"}
    assert f["root_cause"].notna().all() and f["guideline_fix"].notna().all()
    crit = f[f["severity"] == "critical"]
    assert (crit["defect"] == "unsafe").all()
    # every wrong preference really picked the flawed side
    flawed_side = responses[responses["flawed"]].set_index("item_id")["side"]
    wp = f[f["check"] == "wrong_preference"].merge(ratings[["eval_id", "preferred"]], on="eval_id")
    assert (wp["preferred"] == wp["item_id"].map(flawed_side)).all()


def test_audit_sample_is_stratified(round1, cfg):
    _, _, ratings, _, _ = round1
    s = analysis.audit_sample(ratings, cfg)
    per_rater = ratings[ratings["eval_id"].isin(s)].groupby("rater_id").size()
    assert per_rater.nunique() == 1 and len(per_rater) == len(RATERS)


def test_audit_summary_ci_contains_estimate(round1, cfg):
    _, responses, ratings, _, long = round1
    f = analysis.audit_findings(ratings, long, responses)
    summ = analysis.audit_summary(f, analysis.audit_sample(ratings, cfg), len(ratings))
    assert ((summ["ci_low"] <= summ["audit_sample_rate"]) & (summ["audit_sample_rate"] <= summ["ci_high"])).all()


# ---------------------------------------------------------------- LLM judge parsing
def test_parse_judgment():
    text = 'Sure: {"A": {"accuracy": 5, "instruction_following": 5, "relevance": 5, "safety": 5, "clarity": 4},' \
           ' "B": {"accuracy": 1, "instruction_following": 5, "relevance": 5, "safety": 5, "clarity": 5},' \
           ' "preferred": "A", "rationale": "B gives the wrong capital."}'
    j = judge.parse_judgment(text)
    assert j["preferred"] == "A" and j["B"]["accuracy"] == 1
    with pytest.raises(ValueError):
        judge.parse_judgment(text.replace('"accuracy": 1', '"accuracy": 7'))


def test_build_prompt(cfg):
    t = (cfg.prompts_dir / "judge_prompt.txt").read_text()
    p = judge.build_prompt(t, "Q?", "aaa", "bbb")
    assert "Q?" in p and "aaa" in p and "bbb" in p and "{prompt}" not in p
