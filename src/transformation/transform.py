"""Stage 4 - Transformation: derived datasets (train data only) + sparse user-item matrix."""
import logging
import numpy as np
import pandas as pd
import scipy.sparse as sp
from configs import settings as C

log = logging.getLogger(__name__)


def run():
    train = C.load_csv(C.PROC / "train.csv", ["timestamp"])
    items = C.load_csv(C.PROC / "items.csv")
    train = train.assign(positive=(train.rating >= C.REL_THRESHOLD).astype(int))

    ua = train.groupby("user_id").agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"),
                                      first_ts=("timestamp", "min"), last_ts=("timestamp", "max"),
                                      active_days=("timestamp", lambda s: s.dt.date.nunique())).reset_index()
    C.save_csv(ua, C.PROC / "user_activity_summary.csv")

    ip = train.groupby("item_id").agg(n_interactions=("rating", "size"), avg_rating=("rating", "mean"),
                                      n_positive=("positive", "sum")).reset_index()
    ip["popularity_rank"] = ip.n_interactions.rank(ascending=False, method="min").astype(int)
    C.save_csv(ip.sort_values("popularity_rank"), C.PROC / "item_popularity.csv")

    # user-item matrix (values = rating; 0 = unobserved). Index = sorted ids.
    user_ids = np.sort(train.user_id.unique())
    item_ids = np.sort(items.item_id[items.item_id.isin(train.item_id)].unique())
    u2i = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i2i = pd.Series(np.arange(len(item_ids)), index=item_ids)
    R = sp.csr_matrix((train.rating.values.astype(float),
                       (u2i[train.user_id].values, i2i[train.item_id].values)),
                      shape=(len(user_ids), len(item_ids)))
    sp.save_npz(C.PROC / "user_item_matrix.npz", R)
    C.save_csv(pd.DataFrame({"user_id": user_ids}), C.PROC / "matrix_users.csv")
    C.save_csv(pd.DataFrame({"item_id": item_ids}), C.PROC / "matrix_items.csv")
    log.info("matrix %s density=%.3f%%", R.shape, 100 * R.nnz / np.prod(R.shape))


def load_matrix():
    R = sp.load_npz(C.PROC / "user_item_matrix.npz").tocsr()
    u = pd.read_csv(C.PROC / "matrix_users.csv").user_id.values
    i = pd.read_csv(C.PROC / "matrix_items.csv").item_id.values
    return R, u, i
