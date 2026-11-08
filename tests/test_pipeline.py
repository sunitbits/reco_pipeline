"""Run `pytest -q` from the repo root. Rule tests are standalone; data tests need one pipeline run first."""
import numpy as np
import pandas as pd
import pytest
from configs import settings as C
from src.validation.validate import validate_interactions


def _users_items():
    return pd.DataFrame({"user_id": ["U1", "U2"]}), pd.DataFrame({"item_id": ["I1", "I2"]})


def test_validation_flags_bad_rows():
    u, i = _users_items()
    df = pd.DataFrame({
        "user_id": ["U1", "U1", "U2", "U9", "U2", "U1"],
        "item_id": ["I1", "I2", "I1", "I1", "I9", "I1"],
        "rating": ["5", "7", "abc", "4", "3", "4"],
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-01 10:00:00",
                      "2024-01-01 10:00:00", "2024-01-01 10:00:00", "2031-01-01 10:00:00"]})
    errs = []
    clean = validate_interactions(df, u, i, errs)
    reasons = set(pd.concat(errs).error_reason)
    assert len(clean) == 1
    assert {"rating_out_of_range", "invalid_type_rating", "orphan_user_id", "orphan_item_id",
            "timestamp_out_of_range"} <= reasons


@pytest.mark.skipif(not (C.PROC / "train.csv").exists(), reason="run the pipeline first")
def test_time_split_has_no_leakage():
    tr = C.load_csv(C.PROC / "train.csv", ["timestamp"]); te = C.load_csv(C.PROC / "test.csv", ["timestamp"])
    assert tr.timestamp.max() < te.timestamp.min()
    assert set(te.user_id) <= set(tr.user_id) and set(te.item_id) <= set(tr.item_id)


@pytest.mark.skipif(not (C.FEAT / "user_features.csv").exists(), reason="run the pipeline first")
def test_feature_store_cold_start_defaults():
    from src.features.feature_store import FeatureStore
    fs = FeatureStore().load()
    f = fs.get_user_features(["UNKNOWN_USER"])
    assert f.preferred_category.iloc[0] == "unknown" and not f.isna().any().any()


@pytest.mark.skipif(not (C.MODELS / "hybrid_mf.pkl").exists(), reason="run the pipeline first")
def test_predict_known_and_cold_start_user():
    from src.training.predict import recommend
    used, df = recommend("U0001", "hybrid_mf", 5)
    assert used == "hybrid_mf" and len(df) == 5
    used, df = recommend("NEW_USER", "hybrid_mf", 5)
    assert "cold-start" in used and len(df) == 5
