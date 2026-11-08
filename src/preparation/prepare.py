"""Stage 3 - Preparation: clean, standardise, dedupe, impute, and time-based train/test split."""
import json
import logging
from configs import settings as C

log = logging.getLogger(__name__)


def run():
    users = C.load_csv(C.VALID / "users.csv", ["signup_date"])
    items = C.load_csv(C.VALID / "items.csv")
    inter = C.load_csv(C.VALID / "interactions.csv", ["timestamp"])

    # standardise formats
    items["category"] = items.category.str.strip().str.lower()
    items["brand"] = items.brand.fillna("unknown").str.strip().str.lower()
    items["title"] = items.title.str.strip()
    users["country"] = users.country.fillna("UNK").str.strip().str.upper()
    # missing values
    users["age"] = users.age.fillna(users.age.median()).astype(int)
    # duplicates: a user rates an item once -> keep the most recent rating
    before = len(inter)
    inter = (inter.sort_values("timestamp").drop_duplicates(["user_id", "item_id"], keep="last")
             .astype({"rating": int}).reset_index(drop=True))
    log.info("prepare: removed %d repeat user-item rows", before - len(inter))

    # time-based split (no random shuffling -> no leakage from the future)
    cutoff = inter.timestamp.quantile(1 - C.TEST_FRACTION)
    train = inter[inter.timestamp <= cutoff]
    test = inter[inter.timestamp > cutoff]
    n_test = len(test)
    # evaluation needs known users/items (cold-start users/items are reported, not scored)
    test = test[test.user_id.isin(train.user_id) & test.item_id.isin(train.item_id)]
    split = {"cutoff": str(cutoff), "train_rows": len(train), "test_rows": len(test),
             "test_rows_dropped_cold_start": n_test - len(test)}
    (C.REP / "split_report.json").write_text(json.dumps(split, indent=2))
    log.info("prepare split: %s", split)

    for n, d in (("users", users), ("items", items), ("train", train), ("test", test)):
        C.save_csv(d, C.PROC / f"{n}.csv")
