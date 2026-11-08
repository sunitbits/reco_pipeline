"""Stage 7 - Evaluation: Precision@K, Recall@K, NDCG@K, Coverage, Diversity (+ RMSE/MAE)."""
import json
import logging
import pickle
import numpy as np
import pandas as pd
from configs import settings as C
from src.features.feature_store import FeatureStore
from src.training.tracking import Tracker

log = logging.getLogger(__name__)


def content_vectors(fs, item_ids):
    """Item content vectors (category + brand one-hot, standardised log price) for diversity."""
    f = fs.get_item_features(item_ids)
    X = pd.get_dummies(f[["category", "brand"]]).astype(float)
    X["price"] = (np.log(f.price) - np.log(f.price).mean()) / np.log(f.price).std()
    return X.values


def evaluate_model(model, test, fs, k=C.K):
    rel = test[test.rating >= C.REL_THRESHOLD]
    users = np.array(sorted(rel.user_id.unique()))
    users = np.array([u for u in users if u in model.u2i])
    S = model.score_matrix(users).astype(float)
    S[model.seen_mask(users)] = -np.inf                      # never re-recommend seen items
    top = np.argpartition(-S, k, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(S, top, 1), 1), 1)

    rel_sets = rel.groupby("user_id").item_id.apply(lambda s: {model.i2i[i] for i in s if i in model.i2i})
    hits = np.zeros((len(users), k)); n_rel = np.zeros(len(users))
    for r, u in enumerate(users):
        rs = rel_sets[u]; n_rel[r] = len(rs)
        hits[r] = [t in rs for t in top[r]]
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    idcg = np.array([disc[:int(min(n, k))].sum() for n in n_rel])
    V = content_vectors(fs, model.item_ids)[top]            # (n_users, k, d)
    V = V / (np.linalg.norm(V, axis=2, keepdims=True) + 1e-9)
    sim = np.einsum("ukd,ujd->ukj", V, V)
    ild = 1 - (sim.sum((1, 2)) - k) / (k * (k - 1))         # mean pairwise cosine distance

    m = {f"precision@{k}": hits.sum(1).mean() / k,
         f"recall@{k}": (hits.sum(1) / n_rel).mean(),
         f"ndcg@{k}": (dcg / idcg).mean(),
         "coverage": len(np.unique(top)) / len(model.item_ids),
         "diversity": ild.mean()}
    t = test[test.user_id.isin(model.u2i) & test.item_id.isin(model.i2i)]
    err = model.predict_rating(t) - t.rating.values
    m.update({"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()), "n_eval_users": len(users)})
    return {a: float(b) for a, b in m.items()}


def run():
    C.ensure_dirs()
    fs = FeatureStore().load()
    test = C.load_csv(C.PROC / "test.csv", ["timestamp"])
    tr, rows = Tracker(), {}
    for name in ("popularity", "hybrid_mf"):
        model = pickle.loads((C.MODELS / f"{name}.pkl").read_bytes()); model.fs = fs
        m = evaluate_model(model, test, fs)
        rid = json.loads((C.MODELS / f"{name}.run.json").read_text())["run_id"]
        tr.log_metrics(rid, m); rows[name] = m
        tr.end_run(rid)
    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "model"
    C.save_csv(table.reset_index(), C.REP / "model_comparison.csv")
    (C.REP / "model_comparison.md").write_text(table.to_markdown() if _has_tab() else table.to_string())
    log.info("\n%s", table.to_string())


def _has_tab():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False
