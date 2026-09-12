import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from core import FeatureTransform, TemporalState, metrics_from_counts
from model import TemporalAttentionScorer


def test_causal_state_and_directed_score():
    s = TemporalState(mu=0.1)
    s.update(["A", "B"], 0, 100.0)
    s.update(["A"], 1, 200.0)

    assert abs(s.cochange("A", "B") - 0.5) < 1e-9
    assert abs(s.cochange("B", "A") - 1.0) < 1e-9


def test_feature_transform_shape():
    x = np.array(
        [[
            0.5,
            1,
            2,
            1,
            1,
            2,
            np.nan,
            10,
            20,
            np.nan,
            1,
            3,
        ]],
        dtype=np.float32,
    )

    tr = FeatureTransform().fit(x)
    y = tr.transform(x)

    assert y.shape[1] == 18


def test_model_handles_no_history():
    m = TemporalAttentionScorer(
        feature_dim=18,
    )

    cur = torch.zeros((2, 18))
    hist = torch.zeros((2, 20, 18))
    age = torch.zeros((2, 20))
    same = torch.zeros((2, 20))
    mask = torch.zeros(
        (2, 20),
        dtype=torch.bool,
    )

    out = m(
        cur,
        hist,
        age,
        same,
        mask,
    )

    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_metrics():
    m = metrics_from_counts(
        4,
        90,
        6,
        2,
    )

    assert 0 <= m["f1"] <= 1
    assert -1 <= m["mcc"] <= 1
