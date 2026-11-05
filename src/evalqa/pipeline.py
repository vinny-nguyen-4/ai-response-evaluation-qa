"""End-to-end run: simulate a rating round -> agreement -> reviewer scorecards ->
QA audit -> root causes and guideline gaps -> charts, Excel dashboard, memo.

Run with ``python -m evalqa.pipeline``.
"""

from __future__ import annotations

import json

import pandas as pd

from . import analysis, charts, report
from .config import CONFIG, Config
from .simulate import RATERS, load_items, simulate_round


def run(cfg: Config = CONFIG) -> dict:
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    print("[1/5] Simulating a rating round")
    items = load_items(cfg)
    responses, ratings, _ = simulate_round(items, cfg)
    ratings.to_csv(cfg.reports_dir / "ratings_round1.csv", index=False)
    long = analysis.long_scores(ratings, responses)

    print("[2/5] Agreement and reviewer calibration")
    agr = analysis.agreement(long)
    pref_agree = analysis.preference_agreement(ratings)
    sc = analysis.scorecards(long, ratings, responses, cfg)

    print("[3/5] QA audit")
    findings = analysis.audit_findings(ratings, long, responses)
    sample = analysis.audit_sample(ratings, cfg)
    audited = findings[findings["eval_id"].isin(sample)]
    audit_sum = analysis.audit_summary(findings, sample, len(ratings))
    rc = analysis.root_causes(findings)
    placement = analysis.penalty_placement(long)
    audited.to_csv(cfg.reports_dir / "qa_audit_findings.csv", index=False)

    print("[4/5] Charts")
    f = cfg.figures_dir
    imgs = [charts.reviewer_scorecard(sc, cfg.min_detection, cfg.max_abs_bias, f / "reviewer_scorecard.png"),
            charts.detection_heatmap(sc, f / "detection_by_defect.png"),
            charts.agreement_chart(agr, f / "agreement_by_dimension.png"),
            charts.root_cause_chart(rc, f / "root_causes.png")]

    print("[5/5] Excel dashboard and memo")
    a = agr.set_index("dimension")
    pad = placement.set_index("defect").loc["padding"]
    flagged = sc[sc["status"] == "Recalibrate"]
    any_row = audit_sum.iloc[0]
    det_by_type = {d: float(sc[f"caught_{d}"].mean()) for d in analysis.KEY_DIMS}
    s = {
        "items": len(items), "responses": len(responses), "reviewers": len(RATERS),
        "evaluations": len(ratings), "scores": len(long) * 5,
        "alpha": {d: float(a.loc[d, "krippendorff_alpha"]) for d in a.index},
        "preference_agreement": pref_agree,
        "overall_detection": float(sc["detection_rate"].mean()),
        "detection_by_defect": det_by_type,
        "flagged": {r["rater_id"]: r["reasons"] for _, r in flagged.iterrows()},
        "audit_sample": len(sample),
        "audit_rate": float(any_row["audit_sample_rate"]),
        "audit_ci": [float(any_row["ci_low"]), float(any_row["ci_high"])],
        "true_rate": float(any_row["all_evaluations_rate"]),
        "major_rate_all": float(audit_sum.iloc[1]["all_evaluations_rate"]),
        "critical_findings": int((findings["severity"] == "critical").sum()),
        "findings_total": len(findings),
        "top_causes": rc.head(3)[["root_cause", "guideline_fix", "findings"]].to_dict("records"),
        "padding_relevance": float(pad["lowered_relevance"]), "padding_clarity": float(pad["lowered_clarity"]),
    }
    kpis = [("Prompts / responses", f"{s['items']} / {s['responses']}"),
            ("Reviewers / evaluations", f"{s['reviewers']} / {s['evaluations']}"),
            ("Defects caught (avg across reviewers)", f"{s['overall_detection']:.0%}"),
            ("Agreement on preferred response", f"{pref_agree:.0%}"),
            ("Lowest agreement (alpha)", f"{min(s['alpha'], key=s['alpha'].get)} {min(s['alpha'].values()):.2f}"),
            ("Reviewers flagged for recalibration", f"{len(flagged)} of {len(sc)}"),
            ("Evaluations with a QA finding (audit sample)", f"{s['audit_rate']:.0%} (95% CI {s['audit_ci'][0]:.0%}–{s['audit_ci'][1]:.0%})"),
            ("Evaluations with a QA finding (all)", f"{s['true_rate']:.0%}"),
            ("Critical findings (all)", str(s["critical_findings"]))]
    sheets = {
        "Reviewer Scorecards": (sc, "Gold = expert adjudication under v2 guidelines; thresholds in config.py"),
        "Agreement": (agr, "Krippendorff's alpha (interval) across the 4 reviewers per item; 0.667 is a common minimum"),
        "QA Audit Summary": (audit_sum, f"{cfg.audit_share:.0%} stratified audit sample vs. every evaluation"),
        "QA Audit Findings": (audited, "Findings in the audit sample (full list: rerun with all evaluations)"),
        "Root Causes": (rc, "Severity points: critical 3, major 2, minor 1"),
        "Penalty Placement": (placement, "Which dimension reviewers lowered when they caught each defect"),
        "Ratings": (ratings, "Round 1 ratings under v1 guidelines"),
        "Test Set": (responses.drop(columns=[c for c in responses.columns if c.startswith("gold_")]), "Prompts' responses with planted defects"),
    }
    report.build_workbook(cfg.reports_dir / "quality_dashboard.xlsx", sheets, kpis, imgs)
    write_memo(cfg.reports_dir / "findings_memo.md", s, sc)
    (cfg.reports_dir / "summary.json").write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    return s


def write_memo(path, s: dict, sc: pd.DataFrame) -> None:
    names = {"factual_error": "factual errors", "hallucinated_source": "made-up sources",
             "missing_requirement": "missing requirements", "unsafe": "unsafe advice", "padding": "off-topic padding"}
    det = ", ".join(f"{names[k]} {v:.0%}" for k, v in sorted(s["detection_by_defect"].items(), key=lambda kv: kv[1]))
    alpha = ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in sorted(s["alpha"].items(), key=lambda kv: kv[1]))
    flagged = "\n".join(f"| {k} | {v} |" for k, v in s["flagged"].items())
    causes = "\n".join(f"{i + 1}. **{c['root_cause']}** ({c['findings']} findings). Fix: {c['guideline_fix']}."
                       for i, c in enumerate(s["top_causes"]))
    path.write_text(f"""# Rating Quality Review – Round 1 Findings

*Auto-generated by `python -m evalqa.pipeline`. Reviewers are simulated; the test set and guidelines are in this repo.*

## Summary
- {s['reviewers']} reviewers rated {s['items']} prompt pairs ({s['evaluations']} evaluations, {s['scores']:,} scores) under the v1 guidelines.
- Reviewers caught **{s['overall_detection']:.0%}** of planted defects on average. By type: {det}.
- **{len(s['flagged'])} of {s['reviewers']} reviewers** need recalibration.
- The {s['audit_sample']}-evaluation QA audit found problems in **{s['audit_rate']:.0%}** of evaluations (95% CI {s['audit_ci'][0]:.0%}–{s['audit_ci'][1]:.0%}); the rate across all evaluations was {s['true_rate']:.0%}, so the sample estimate held up. {s['critical_findings']} findings were critical (unsafe advice missed or preferred).

## Agreement
- Krippendorff's alpha by dimension: {alpha}. Reviewers agreed on the preferred response {s['preference_agreement']:.0%} of the time.
- Relevance and clarity agreement is low even though nearly every reviewer caught the padding: **{s['padding_relevance']:.0%} lowered Relevance and {s['padding_clarity']:.0%} lowered Clarity**. That's a guideline gap, not a skill gap — v1 never says where off-topic content belongs.

## Reviewers flagged for recalibration

| Reviewer | Why |
|---|---|
{flagged}

## Top root causes
{causes}

## Guideline changes (v2)
See [`guidelines/rating_guidelines_v2.md`](../guidelines/rating_guidelines_v2.md):
1. Verify every fact and source before scoring Accuracy; an unverifiable source counts as an error.
2. Off-topic padding lowers Relevance, not Clarity.
3. List the prompt's requirements before scoring Instruction following.
4. An unsafe response can never be preferred over a safe one.
5. Score both responses before choosing; the preferred one must score higher.
6. Every problem named in a rationale must lower a score.

## Next steps
- Run a calibration session with the flagged reviewers on the examples where they missed defects (the audit file lists them).
- Re-rate a fresh sample under v2 and compare agreement on Relevance and Clarity against this round.

## Limitations
- Reviewers are simulated with set habits, so results show the QA method, not real people's performance.
- 32 prompts and 16 evaluations per reviewer: per-reviewer rates for rare defect types (4 unsafe prompts) rest on very few ratings.
""")


if __name__ == "__main__":
    run()
