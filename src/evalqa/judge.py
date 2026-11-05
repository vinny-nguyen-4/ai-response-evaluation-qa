"""Optional LLM-as-judge: score the same pairs with Claude using the v2 rubric, then
compare the judge with the gold scores like any other reviewer.

Runs only when ANTHROPIC_API_KEY is set and the ``anthropic`` package is installed:
    pip install anthropic && ANTHROPIC_API_KEY=... python -m evalqa.judge
"""

from __future__ import annotations

import json
import os

import pandas as pd

from .config import CONFIG, DIMENSIONS, Config


def build_prompt(template: str, prompt: str, a: str, b: str) -> str:
    return template.replace("{prompt}", prompt).replace("{response_a}", a).replace("{response_b}", b)


def parse_judgment(text: str) -> dict:
    """Parse and validate the judge's JSON reply."""
    data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    for side in "AB":
        for d in DIMENSIONS:
            v = int(data[side][d])
            if not 1 <= v <= 5:
                raise ValueError(f"{side}.{d} out of range: {v}")
            data[side][d] = v
    if data["preferred"] not in ("A", "B"):
        raise ValueError("preferred must be A or B")
    return data


def judge_round(items: pd.DataFrame, responses: pd.DataFrame, cfg: Config = CONFIG,
                model: str = "claude-sonnet-5-5") -> pd.DataFrame:  # pragma: no cover - network
    import anthropic

    client = anthropic.Anthropic()
    template = (cfg.prompts_dir / "judge_prompt.txt").read_text()
    rows = []
    for _, it in items.iterrows():
        r = responses[responses["item_id"] == it["id"]].set_index("side")
        msg = client.messages.create(model=model, max_tokens=800, messages=[
            {"role": "user", "content": build_prompt(template, it["prompt"], r.loc["A", "text"], r.loc["B", "text"])}])
        j = parse_judgment(msg.content[0].text)
        rows.append({"eval_id": f"{it['id']}-JUDGE", "item_id": it["id"], "rater_id": "JUDGE",
                     **{f"{s}_{d}": j[s][d] for s in "AB" for d in DIMENSIONS},
                     "preferred": j["preferred"], "rationale": j["rationale"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":  # pragma: no cover
    from . import analysis
    from .simulate import load_items, simulate_round

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the LLM judge.")
    items = load_items()
    responses, _, _ = simulate_round(items)
    judged = judge_round(items, responses)
    long = analysis.long_scores(judged, responses)
    print(analysis.scorecards(long, judged, responses).to_string())
    judged.to_csv(CONFIG.reports_dir / "llm_judge_ratings.csv", index=False)
