"""Stage 6 - Models. Both expose the same interface so evaluation is model-agnostic:
    fit() | score_matrix(user_ids) -> (n_users x n_items) | predict_rating(df)"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from configs import settings as C
from src.transformation.transform import load_matrix


class BaseRecommender:
    name = "base"

    def _load(self):
        self.R, self.user_ids, self.item_ids = load_matrix()
        self.u2i = {u: i for i, u in enumerate(self.user_ids)}
        self.i2i = {i: j for j, i in enumerate(self.item_ids)}

    def rows(self, users):
        return np.array([self.u2i[u] for u in users])

    def seen_mask(self, users):
        return self.R[self.rows(users)].toarray() > 0

    def params(self):
        return {}


class PopularityRecommender(BaseRecommender):
    """Baseline: same ranking for everyone = number of positive (>=4) ratings.
    Rating prediction = smoothed item mean."""
    name = "popularity"

    def __init__(self, fs):
        self.fs = fs

    def fit(self):
        self._load()
        itf = self.fs.get_item_features(self.item_ids)
        self.scores = itf.item_positive_count.values.astype(float)
        self.item_avg = itf.item_avg_rating.values
        return self

    def score_matrix(self, users):
        return np.tile(self.scores, (len(users), 1))

    def predict_rating(self, df):
        return self.item_avg[df.item_id.map(self.i2i).values]

    def params(self):
        return {"model": "popularity", "ranking_signal": "item_positive_count"}


def _z(a):
    return (a - a.mean(axis=-1, keepdims=True)) / (a.std(axis=-1, keepdims=True) + 1e-9)


class HybridMFRecommender(BaseRecommender):
    """Personalised hybrid.
    Rating model: user mean + shrunken item bias + low-rank (truncated SVD) residual factors.
    Ranking score: z(predicted rating) + alpha * z(log item popularity) + beta * z(user-category affinity)
    (the last two terms come from the feature store)."""
    name = "hybrid_mf"

    def __init__(self, fs, n_factors=16, alpha=2.0, beta=1.0, item_shrink=10, seed=C.SEED):
        self.fs, self.n_factors, self.alpha, self.beta = fs, n_factors, alpha, beta
        self.item_shrink, self.seed = item_shrink, seed

    def fit(self):
        self._load()
        R = self.R.tocoo()
        n_u, n_i = R.shape
        cnt = np.maximum(np.diff(self.R.indptr), 1)
        self.mu = np.asarray(self.R.sum(1)).ravel() / cnt
        res = R.data - self.mu[R.row]
        ic = np.bincount(R.col, minlength=n_i)
        self.b_i = np.bincount(R.col, weights=res, minlength=n_i) / (ic + self.item_shrink)
        X = csr_matrix((res - self.b_i[R.col], (R.row, R.col)), shape=(n_u, n_i))
        svd = TruncatedSVD(self.n_factors, n_iter=10, random_state=self.seed)
        self.Uf, self.Vt = svd.fit_transform(X), svd.components_
        self.explained_var = float(svd.explained_variance_ratio_.sum())
        # content/feature-store signals
        itf = self.fs.get_item_features(self.item_ids)
        self.pop_z = _z(np.log1p(itf.item_popularity.values.astype(float)))
        cats = sorted(itf.category.unique())
        self.item_cat = itf.category.map({c: k for k, c in enumerate(cats)}).values
        ucf = self.fs.user_cat.pivot(index="user_id", columns="category", values="user_category_avg_rating")
        A = ucf.reindex(index=self.user_ids, columns=cats)
        uavg = self.fs.get_user_features(self.user_ids).user_avg_rating.values
        self.A = A.apply(lambda col: col.fillna(pd.Series(uavg, index=A.index))).values
        return self

    def _pred(self, rows):
        return self.mu[rows, None] + self.b_i[None, :] + self.Uf[rows] @ self.Vt

    def score_matrix(self, users):
        rows = self.rows(users)
        cat_aff = self.A[rows][:, self.item_cat]
        return _z(self._pred(rows)) + self.alpha * self.pop_z[None, :] + self.beta * _z(cat_aff)

    def predict_rating(self, df):
        u = self.rows(df.user_id.values); i = df.item_id.map(self.i2i).values
        p = self.mu[u] + self.b_i[i] + (self.Uf[u] * self.Vt[:, i].T).sum(1)
        return np.clip(p, 1, 5)

    def params(self):
        return {"model": "hybrid_mf", "n_factors": self.n_factors, "alpha_popularity": self.alpha,
                "beta_category": self.beta, "item_shrink": self.item_shrink, "seed": self.seed}
