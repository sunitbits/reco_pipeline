"""Stage 1 - Ingestion. Generates DOCUMENTED SYNTHETIC e-commerce data and writes raw CSVs.

Synthetic design
- 1500 users, 600 items, 8 categories, ~60k rating events (1-5) over 2023-2024.
- Ratings come from latent user/item factors (+ category structure, item bias, noise),
  so personalised models have real signal to learn.
- Exposure is popularity-biased (users mostly interact with popular items), like real data.
- Deliberate dirt is injected (nulls, dupes, bad ratings/timestamps/types, orphan keys)
  so the validation stage has something to catch. Raw files are all-strings, as a
  real CSV drop would be.
To use real data (e.g. Amazon Reviews), replace `generate_synthetic` with a loader that
returns the same three frames: users, items, interactions.
"""
import logging
import numpy as np
import pandas as pd
from configs import settings as C

log = logging.getLogger(__name__)
CATS = ["electronics", "books", "home", "sports", "beauty", "toys", "grocery", "fashion"]


def generate_synthetic(seed=C.SEED, n_users=1500, n_items=600):
    rng = np.random.default_rng(seed)
    D = 8
    cat_idx = rng.integers(0, len(CATS), n_items)
    items = pd.DataFrame({
        "item_id": [f"I{i:04d}" for i in range(n_items)],
        "title": [f"Item {i}" for i in range(n_items)],
        "category": np.array(CATS)[cat_idx],
        "brand": [f"brand_{b}" for b in rng.integers(0, 12, n_items)],
        "price": np.round(np.exp(rng.normal(3.5, 0.8, n_items)), 2),
    })
    start = pd.Timestamp(C.DATA_START)
    signup = start + pd.to_timedelta(rng.integers(-300, 500, n_users), unit="D")
    users = pd.DataFrame({
        "user_id": [f"U{i:04d}" for i in range(n_users)],
        "age": rng.integers(18, 70, n_users),
        "country": rng.choice(["IN", "US", "GB", "DE", "AE"], n_users),
        "signup_date": signup.strftime("%Y-%m-%d"),
    })
    cat_emb = rng.normal(size=(len(CATS), D))
    V = cat_emb[cat_idx] * 0.8 + rng.normal(size=(n_items, D)) * 0.8
    U = rng.normal(size=(n_users, D))
    aff = U @ V.T / np.sqrt(D)
    item_bias = rng.normal(0, 0.4, n_items)
    pop = rng.pareto(1.2, n_items) + 1
    pop /= pop.sum()

    end = pd.Timestamp(C.DATA_END)
    rows = []
    for u in range(n_users):
        n_u = int(np.clip(rng.lognormal(np.log(30), 0.8), 5, 150))
        p = pop * np.exp(1.0 * aff[u]); p /= p.sum()
        chosen = rng.choice(n_items, size=n_u, replace=False, p=p)
        lo = max(signup[u], start)
        span = (end - lo).total_seconds()
        ts = lo + pd.to_timedelta(np.round(rng.random(n_u) * span), unit="s")
        r = 3.4 + 0.9 * aff[u, chosen] + item_bias[chosen] + rng.normal(0, 0.5, n_u)
        r = np.clip(np.round(r), 1, 5).astype(int)
        rows.append(pd.DataFrame({"user_id": users.user_id[u], "item_id": items.item_id[chosen],
                                  "rating": r, "timestamp": ts}))
    inter = pd.concat(rows, ignore_index=True)
    inter["timestamp"] = inter.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
    return users, items, inter


def _inject_noise(users, items, inter, seed):
    """Return all-string frames with known dirt injected."""
    rng = np.random.default_rng(seed + 1)
    users, items, inter = (d.astype(str).copy() for d in (users, items, inter))
    n = len(inter)
    perm = iter(np.array_split(rng.permutation(n), 200))  # 0.5% chunks
    def take(k):
        return np.concatenate([next(perm) for _ in range(k)])
    inter.loc[take(1) , "rating"] = np.nan                       # missing rating (0.5%)
    idx = take(1); inter.loc[idx, "rating"] = rng.choice(["0", "6", "-1", "11"], len(idx))  # out of range
    inter.loc[take(1), "rating"] = "abc"                          # wrong type
    inter.loc[take(1), "timestamp"] = "2031-05-01 00:00:00"       # future
    inter.loc[take(1), "timestamp"] = "not_a_date"                # unparsable
    inter.loc[take(1), "user_id"] = "U9999"                       # orphan user
    inter.loc[take(1), "item_id"] = "I9999"                       # orphan item
    inter.loc[take(1), "user_id"] = np.nan                        # missing key
    inter = pd.concat([inter, inter.sample(frac=0.01, random_state=seed)], ignore_index=True)  # dupes

    ui = rng.permutation(len(users))
    users.loc[ui[:8], "age"] = np.nan                             # missing age (allowed; imputed)
    users.loc[ui[8:12], "age"] = "250"                            # impossible age
    users.loc[ui[12:15], "signup_date"] = "31/31/2023"            # bad date
    users = pd.concat([users, users.iloc[ui[15:20]]], ignore_index=True)  # dupes
    ii = rng.permutation(len(items))
    items.loc[ii[:4], "category"] = np.nan
    items.loc[ii[4:7], "price"] = "-5"
    items.loc[ii[7:9], "price"] = "free"
    items = pd.concat([items, items.iloc[ii[9:12]]], ignore_index=True)
    return users, items, inter


def run():
    C.ensure_dirs()
    users, items, inter = generate_synthetic()
    users, items, inter = _inject_noise(users, items, inter, C.SEED)
    for name, df in (("users", users), ("items", items), ("interactions", inter)):
        C.save_csv(df, C.RAW / f"{name}.csv")
        log.info("ingested %-12s rows=%d", name, len(df))
