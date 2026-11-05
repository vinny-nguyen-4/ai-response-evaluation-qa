"""Quality analysis of a rating round: agreement, reviewer calibration, QA audit,
root causes and guideline gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CONFIG, DIMENSIONS, Config

KEY_DIMS = {"factual_error": ["accuracy"], "hallucinated_source": ["accuracy"],
            "missing_requirement": ["instruction_following"], "unsafe": ["safety"],
            "padding": ["relevance", "clarity"]}

ROOT_CAUSE = {
    ("missed_defect", "factual_error"): ("v1 has no step requiring reviewers to verify facts and math", "Rule 1: verify before scoring Accuracy"),
    ("missed_defect", "hallucinated_source"): ("v1 doesn't say how to score a source that can't be verified", "Rule 1 + Accuracy anchor: unverifiable source = Accuracy 2 or lower"),
    ("missed_defect", "missing_requirement"): ("v1 doesn't ask reviewers to list the prompt's requirements", "Rule 3: count the requirements"),
    ("missed_defect", "unsafe"): ("Reviewer missed harmful advice; safety calibration gap", "Recalibration session on safety examples"),
    ("missed_defect", "padding"): ("Off-topic content not recognized as a problem", "Relevance anchor for 3 clarified"),
    ("wrong_preference", "unsafe"): ("v1 lets an unsafe response win a preference", "Rule 4: safety overrides preference"),
    ("wrong_preference", None): ("v1 doesn't tie the preferred response to the scores", "Rule 5: score both responses first; the preferred one must score higher"),
    ("rationale_mismatch", None): ("v1 doesn't require rationale and scores to agree", "Rationale rule: every problem named must lower a score"),
    ("misapplied_dimension", "padding"): ("v1 doesn't say whether padding is a Relevance or Clarity problem", "Rule 2: padding is a Relevance problem"),
    ("false_flag", None): ("Good response scored 2 or lower with no defect", "Anchor examples for 5-point scores"),
}
SEVERITY_WEIGHT = {"critical": 3, "major": 2, "minor": 1}


# ---------------------------------------------------------------- reshaping
def long_scores(ratings: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """One row per rater x response, with the rater's scores and the gold scores."""
    parts = []
    for side in "AB":
        cols = {f"{side}_{d}": d for d in DIMENSIONS}
        p = ratings[["eval_id", "item_id", "rater_id", *cols]].rename(columns=cols).assign(side=side)
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    return long.merge(responses.drop(columns=["text"]), on=["item_id", "side"])


# ---------------------------------------------------------------- agreement
def krippendorff_alpha_interval(units: list[list[float]]) -> float:
    """Krippendorff's alpha with the interval distance. ``units`` = list of value lists,
    one per rated unit; units with fewer than two ratings are ignored."""
    units = [np.asarray(u, float) for u in units if len(u) >= 2]
    if not units:
        return float("nan")
    allv = np.concatenate(units)
    n = len(allv)
    d_o = sum(((u[:, None] - u[None, :]) ** 2).sum() / (len(u) - 1) for u in units) / n
    d_e = ((allv[:, None] - allv[None, :]) ** 2).sum() / (n * (n - 1))
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def agreement(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in DIMENSIONS:
        units = [g[d].tolist() for _, g in long.groupby(["item_id", "side"])]
        within1, exact, pairs = 0, 0, 0
        for u in units:
            for i in range(len(u)):
                for j in range(i + 1, len(u)):
                    pairs += 1
                    exact += u[i] == u[j]
                    within1 += abs(u[i] - u[j]) <= 1
        rows.append({"dimension": d, "krippendorff_alpha": krippendorff_alpha_interval(units),
                     "exact_agreement": exact / pairs, "within_1_agreement": within1 / pairs})
    return pd.DataFrame(rows)


def preference_agreement(ratings: pd.DataFrame) -> float:
    """Share of rater pairs on the same item that chose the same response."""
    agree = total = 0
    for _, g in ratings.groupby("item_id"):
        p = g["preferred"].tolist()
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                total += 1
                agree += p[i] == p[j]
    return agree / total


# ---------------------------------------------------------------- reviewer calibration
def caught(row: pd.Series) -> bool:
    """A reviewer caught the defect if they scored any of its key dimensions 3 or lower."""
    return any(row[d] <= 3 for d in KEY_DIMS[row["defect"]])


def scorecards(long: pd.DataFrame, ratings: pd.DataFrame, responses: pd.DataFrame, cfg: Config = CONFIG) -> pd.DataFrame:
    gold_cols = [f"gold_{d}" for d in DIMENSIONS]
    diff = long[DIMENSIONS].to_numpy() - long[gold_cols].to_numpy()
    long = long.assign(abs_err=np.abs(diff).mean(axis=1), err=diff.mean(axis=1))
    flawed = long[long["flawed"]].copy()
    flawed["caught"] = flawed.apply(caught, axis=1)
    good_side = responses[~responses["flawed"]].set_index("item_id")["side"]
    pref_ok = ratings["preferred"] == ratings["item_id"].map(good_side)
    rows = []
    for rid, g in long.groupby("rater_id"):
        f = flawed[flawed["rater_id"] == rid]
        r = ratings[ratings["rater_id"] == rid]
        sc = {"rater_id": rid, "evaluations": len(r), "mae_vs_gold": g["abs_err"].mean(), "bias": g["err"].mean(),
              "detection_rate": f["caught"].mean(), "preference_accuracy": pref_ok[r.index].mean()}
        for dft in KEY_DIMS:
            fd = f[f["defect"] == dft]
            sc[f"caught_{dft}"] = fd["caught"].mean() if len(fd) else np.nan
        reasons = []
        if sc["mae_vs_gold"] > cfg.max_mae:
            reasons.append(f"error {sc['mae_vs_gold']:.2f} > {cfg.max_mae}")
        if abs(sc["bias"]) > cfg.max_abs_bias:
            reasons.append(f"{'lenient' if sc['bias'] > 0 else 'harsh'} by {abs(sc['bias']):.2f} pts")
        if sc["detection_rate"] < cfg.min_detection:
            reasons.append(f"caught {sc['detection_rate']:.0%} of defects")
        if sc["caught_unsafe"] < 1:
            reasons.append(f"missed unsafe advice ({sc['caught_unsafe']:.0%} caught)")
        if sc["preference_accuracy"] < cfg.min_pref_accuracy:
            reasons.append(f"picked the better response {sc['preference_accuracy']:.0%} of the time")
        sc["status"] = "Recalibrate" if reasons else "On target"
        sc["reasons"] = "; ".join(reasons)
        rows.append(sc)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- QA audit
def audit_findings(ratings: pd.DataFrame, long: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Every QA check, applied to every evaluation. The audit samples from this."""
    meta = responses[responses["flawed"]].set_index("item_id")[["side", "defect"]]
    findings = []
    lk = long.set_index(["eval_id", "side"])
    for _, r in ratings.iterrows():
        fside, defect = meta.loc[r["item_id"], "side"], meta.loc[r["item_id"], "defect"]
        gside = "B" if fside == "A" else "A"
        f = lk.loc[(r["eval_id"], fside)]
        g = lk.loc[(r["eval_id"], gside)]
        names_problem = "no significant issues" not in r["rationale"]
        is_caught = any(f[d] <= 3 for d in KEY_DIMS[defect])
        base = {"eval_id": r["eval_id"], "item_id": r["item_id"], "rater_id": r["rater_id"], "defect": defect}
        if not is_caught and not names_problem:
            findings.append({**base, "check": "missed_defect", "severity": "critical" if defect == "unsafe" else "major"})
        if names_problem and not is_caught:
            findings.append({**base, "check": "rationale_mismatch", "severity": "minor"})
        if r["preferred"] == fside:
            findings.append({**base, "check": "wrong_preference", "severity": "critical" if defect == "unsafe" else "major"})
        if defect == "padding" and f["clarity"] <= 3 and f["relevance"] >= 4:
            findings.append({**base, "check": "misapplied_dimension", "severity": "minor"})
        if min(g[d] for d in DIMENSIONS) <= 2:
            findings.append({**base, "check": "false_flag", "severity": "minor"})
    out = pd.DataFrame(findings, columns=["eval_id", "item_id", "rater_id", "defect", "check", "severity"])
    causes = [ROOT_CAUSE.get((c, d)) or ROOT_CAUSE.get((c, None)) for c, d in zip(out["check"], out["defect"])]
    out["root_cause"] = [c[0] for c in causes]
    out["guideline_fix"] = [c[1] for c in causes]
    return out


def audit_sample(ratings: pd.DataFrame, cfg: Config = CONFIG) -> pd.Series:
    """Stratified random sample of evaluations (same share from every reviewer)."""
    rng = np.random.default_rng(cfg.seed + 1)
    ids = []
    for _, g in ratings.groupby("rater_id"):
        k = max(1, round(len(g) * cfg.audit_share))
        ids += list(rng.choice(g["eval_id"].to_numpy(), size=k, replace=False))
    return pd.Series(sorted(ids), name="eval_id")


def audit_summary(findings: pd.DataFrame, sample_ids: pd.Series, n_total: int) -> pd.DataFrame:
    """Error rate estimated from the audit sample vs. the true rate across all evaluations."""
    rows = []
    s = findings[findings["eval_id"].isin(sample_ids)]
    n_s = len(sample_ids)
    for label, df_s, df_all in [("Any finding", s, findings),
                                ("Major or critical", s[s["severity"] != "minor"], findings[findings["severity"] != "minor"]),
                                ("Critical", s[s["severity"] == "critical"], findings[findings["severity"] == "critical"])]:
        p = df_s["eval_id"].nunique() / n_s
        se = np.sqrt(p * (1 - p) / n_s)
        rows.append({"measure": f"Evaluations with: {label}", "audit_sample_rate": p,
                     "ci_low": max(0.0, p - 1.96 * se), "ci_high": min(1.0, p + 1.96 * se),
                     "all_evaluations_rate": df_all["eval_id"].nunique() / n_total})
    return pd.DataFrame(rows)


def root_causes(findings: pd.DataFrame) -> pd.DataFrame:
    f = findings.assign(weight=findings["severity"].map(SEVERITY_WEIGHT))
    g = f.groupby(["root_cause", "guideline_fix"]).agg(findings=("eval_id", "count"),
                                                         critical=("severity", lambda s: (s == "critical").sum()),
                                                         severity_points=("weight", "sum"))
    return g.sort_values("severity_points", ascending=False).reset_index()


# ---------------------------------------------------------------- guideline gaps
def penalty_placement(long: pd.DataFrame) -> pd.DataFrame:
    """For caught defects, which dimension did reviewers lower? A split means the
    guideline is ambiguous, not that reviewers lack skill."""
    f = long[long["flawed"]].copy()
    f = f[f.apply(caught, axis=1)]
    rows = []
    for dft, g in f.groupby("defect"):
        lowered = {d: (g[d] <= 3).mean() for d in DIMENSIONS}
        expected = KEY_DIMS[dft][0]
        rows.append({"defect": dft, "caught_ratings": len(g), "expected_dimension": expected,
                     "share_on_expected": lowered[expected],
                     **{f"lowered_{d}": v for d, v in lowered.items()}})
    return pd.DataFrame(rows).sort_values("share_on_expected")
