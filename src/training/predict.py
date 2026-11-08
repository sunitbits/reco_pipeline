"""Prediction path: recommendations for ONE user from a trained model + the shared feature store
(the same feature layer used in training). Unknown users fall back to the popularity model.
    python -m src.training.predict U0001 --model hybrid_mf --k 10"""
import argparse
import pickle
import numpy as np
from configs import settings as C
from src.features.feature_store import FeatureStore


def _load(name, fs):
    m = pickle.loads((C.MODELS / f"{name}.pkl").read_bytes()); m.fs = fs
    return m


def recommend(user_id, model_name="hybrid_mf", k=C.K):
    fs = FeatureStore().load()
    model = _load(model_name, fs)
    if user_id in model.u2i:
        s = model.score_matrix([user_id])[0].astype(float)
        s[model.seen_mask([user_id])[0]] = -np.inf           # never re-recommend seen items
        used = model_name
    else:                                                    # cold start: no history for this user
        model = _load("popularity", fs)
        s, used = model.scores.astype(float), "popularity (cold-start fallback)"
    top = np.argsort(-s)[:k]
    out = fs.get_item_features(model.item_ids[top])[["category", "brand", "price", "item_avg_rating", "item_popularity"]].copy()
    out.insert(0, "score", s[top].round(3))
    out.insert(0, "rank", range(1, len(top) + 1))
    return used, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("user_id"); ap.add_argument("--model", default="hybrid_mf"); ap.add_argument("--k", type=int, default=C.K)
    a = ap.parse_args()
    used, df = recommend(a.user_id, a.model, a.k)
    print(f"Top-{a.k} for {a.user_id} using {used}"); print(df.to_string())
