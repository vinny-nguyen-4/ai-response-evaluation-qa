import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evalqa import analysis  # noqa: E402
from evalqa.config import CONFIG  # noqa: E402
from evalqa.simulate import load_items, simulate_round  # noqa: E402


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    return replace(CONFIG, reports_dir=tmp_path_factory.mktemp("reports"))


@pytest.fixture(scope="session")
def round1(cfg):
    items = load_items(cfg)
    responses, ratings, truth = simulate_round(items, cfg)
    long = analysis.long_scores(ratings, responses)
    return items, responses, ratings, truth, long
