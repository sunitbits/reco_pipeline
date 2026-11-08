"""Stage 5 - Feature management: a small custom feature store (offline, file-backed).

* build()  -> computes user / item / user-category feature tables from TRAIN data only,
              as of `ref_time` (= last training timestamp), and writes a registry (docs).
* get_*()  -> the single read path used by BOTH training and prediction, so features
              are identical online/offline. Unknown (cold-start) entities get defaults.
"""
import json
import logging
import numpy as np
import pandas as pd
from configs import settings as C

log = logging.getLogger(__name__)
SMOOTH_M = 10  # Bayesian smoothing strength for item average rating

REGISTRY = {
    "user_activity_count": ("user", "int", "Number of training interactions"),
    "user_avg_rating": ("user", "float", "Mean rating given by the user"),
    "user_rating_std": ("user", "float", "Std-dev of the user's ratings (0 if single rating)"),
    "user_positive_ratio": ("user", "float", f"Share of the user's ratings >= {C.REL_THRESHOLD}"),
    "preferred_category": ("user", "str", "Category with most interactions"),
    "days_since_last_interaction": ("user", "int", "Days from user's last interaction to ref_time"),
    "user_tenure_days": ("user", "int", "Days from signup to ref_time"),
    "item_popularity": ("item", "int", "Number of training interactions"),
    "item_positive_count": ("item", "int", f"Number of ratings >= {C.REL_THRESHOLD}"),
    "item_popularity_pct": ("item", "float", "Percentile rank of popularity (1 = most popular)"),
    "item_avg_rating": ("item", "float", f"Bayesian-smoothed mean rating (m={SMOOTH_M}) toward global mean"),
    "item_days_since_last_interaction": ("item", "int", "Days since the item was last interacted with"),
    "category / brand / price": ("item", "str/float", "Item content attributes"),
    "user_category_avg_rating": ("user x category", "float", "User's mean rating within a category"),
    "user_category_count": ("user x category", "int", "User's interactions within a category"),
    "user_category_affinity (derived)": ("interaction", "float", "user_category_avg_rating for the item's category, else user_avg_rating"),
    "rating_gap (derived)": ("interaction", "float", "item_avg_rating - user_avg_rating"),
    "is_preferred_category (derived)": ("interaction", "int", "1 if item category == user's preferred_category"),
}


class FeatureStore:
    def __init__(self, path=C.FEAT):
        self.path = path
        self._loaded = False

    # ---------------------------------------------------------------- build
    def build(self):
        train = C.load_csv(C.PROC / "train.csv", ["timestamp"])
        users = C.load_csv(C.PROC / "users.csv", ["signup_date"])
        items = C.load_csv(C.PROC / "items.csv")
        ref = train.timestamp.max()
        train = train.assign(pos=(train.rating >= C.REL_THRESHOLD).astype(int)).merge(items[["item_id", "category"]], on="item_id")
        gmean = train.rating.mean()

        g = train.groupby("user_id")
        uf = g.agg(user_activity_count=("rating", "size"), user_avg_rating=("rating", "mean"),
                   user_rating_std=("rating", "std"), user_positive_ratio=("pos", "mean"),
                   last=("timestamp", "max")).reset_index()
        uf["user_rating_std"] = uf.user_rating_std.fillna(0)
        uf["days_since_last_interaction"] = (ref - uf.pop("last")).dt.days
        cat_counts = train.groupby(["user_id", "category"]).size().reset_index(name="n")
        pref = cat_counts.sort_values(["user_id", "n", "category"], ascending=[True, False, True]).drop_duplicates("user_id")
        uf = uf.merge(pref[["user_id", "category"]].rename(columns={"category": "preferred_category"}), on="user_id")
        uf = uf.merge(users[["user_id", "signup_date"]], on="user_id", how="left")
        uf["user_tenure_days"] = (ref - uf.pop("signup_date")).dt.days

        ig = train.groupby("item_id").agg(item_popularity=("rating", "size"), item_sum=("rating", "sum"),
                                          item_positive_count=("pos", "sum"), last=("timestamp", "max")).reset_index()
        itf = items.merge(ig, on="item_id", how="left")
        itf[["item_popularity", "item_sum", "item_positive_count"]] = itf[["item_popularity", "item_sum", "item_positive_count"]].fillna(0)
        itf["item_avg_rating"] = (itf.item_sum + SMOOTH_M * gmean) / (itf.item_popularity + SMOOTH_M)
        itf["item_popularity_pct"] = itf.item_popularity.rank(pct=True)
        itf["item_days_since_last_interaction"] = (ref - itf.pop("last")).dt.days.fillna(9999).astype(int)
        itf = itf.drop(columns=["item_sum", "title"])

        ucf = train.groupby(["user_id", "category"]).agg(user_category_avg_rating=("rating", "mean"),
                                                         user_category_count=("rating", "size")).reset_index()
        defaults = {"global_mean": gmean, "user_activity_count": 0, "user_avg_rating": gmean, "user_rating_std": 0.0,
                    "user_positive_ratio": float((train.rating >= C.REL_THRESHOLD).mean()),
                    "days_since_last_interaction": 9999, "user_tenure_days": 0}
        C.save_csv(uf, self.path / "user_features.csv")
        C.save_csv(itf, self.path / "item_features.csv")
        C.save_csv(ucf, self.path / "user_category_features.csv")
        meta = {"ref_time": str(ref), "defaults": defaults,
                "registry": {k: {"entity": e, "dtype": d, "description": desc} for k, (e, d, desc) in REGISTRY.items()}}
        (self.path / "feature_registry.json").write_text(json.dumps(meta, indent=2))
        self._write_catalog()
        log.info("features: users=%d items=%d user_cat=%d", len(uf), len(itf), len(ucf))
        return self

    @staticmethod
    def _write_catalog():
        """Feature catalog as a markdown table (reports/feature_catalog.md)."""
        rows = ["# Feature catalog", "",
                "All features are computed from TRAIN data only, as of the last training timestamp. "
                "Cold-start users/items get defaults from the registry.", "",
                "| Feature | Entity | Type | Description |", "|---|---|---|---|"]
        rows += [f"| `{k}` | {e} | {d} | {desc} |" for k, (e, d, desc) in REGISTRY.items()]
        (C.REP / "feature_catalog.md").write_text("\n".join(rows) + "\n")

    # ----------------------------------------------------------------- read
    def load(self):
        self.user = pd.read_csv(self.path / "user_features.csv").set_index("user_id")
        self.item = pd.read_csv(self.path / "item_features.csv").set_index("item_id")
        self.user_cat = pd.read_csv(self.path / "user_category_features.csv")
        self.meta = json.loads((self.path / "feature_registry.json").read_text())
        self._loaded = True
        return self

    def get_user_features(self, ids):
        f = self.user.reindex(ids)
        d = self.meta["defaults"]
        for c in d:
            if c in f.columns:
                f[c] = f[c].fillna(d[c])
        f["preferred_category"] = f.preferred_category.fillna("unknown")
        return f

    def get_item_features(self, ids):
        f = self.item.reindex(ids)
        f["item_avg_rating"] = f.item_avg_rating.fillna(self.meta["defaults"]["global_mean"])
        return f

    def interaction_features(self, pairs):
        """pairs: DataFrame[user_id, item_id] -> row-aligned derived features (training or serving)."""
        uf = self.get_user_features(pairs.user_id.values).reset_index(drop=True)
        itf = self.get_item_features(pairs.item_id.values).reset_index(drop=True)
        out = pairs[["user_id", "item_id"]].reset_index(drop=True)
        out["category"] = itf.category
        out = out.merge(self.user_cat.rename(columns={"user_category_avg_rating": "_ucar"})[["user_id", "category", "_ucar"]],
                        on=["user_id", "category"], how="left")
        out["user_category_affinity"] = out._ucar.fillna(uf.user_avg_rating)
        out["rating_gap"] = itf.item_avg_rating - uf.user_avg_rating
        out["is_preferred_category"] = (itf.category == uf.preferred_category).astype(int)
        return out.drop(columns=["_ucar", "category"])
