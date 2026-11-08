"""Stage 2 - Validation. Rule-based checks (Pandera/Great Expectations could replace these).
Bad rows are removed from the clean tables and written to data/errors/rejected_records.csv
with a reason. A quality gate fails the run if too much data is rejected."""
import json
import logging
import numpy as np
import pandas as pd
from configs import settings as C

log = logging.getLogger(__name__)


def _drop(df, mask, reason, errs, table):
    mask = mask.reindex(df.index, fill_value=False)
    if mask.any():
        bad = df.loc[mask]
        errs.append(pd.DataFrame({"source_table": table, "error_reason": reason,
                                  "record": [json.dumps(r, default=str) for r in bad.to_dict("records")]}))
    return df.loc[~mask]


def validate_users(df, errs):
    t = "users"
    df = _drop(df, df.user_id.isna(), "missing_user_id", errs, t)
    df = _drop(df, df.duplicated("user_id"), "duplicate_user_id", errs, t)
    sd = pd.to_datetime(df.signup_date, format="%Y-%m-%d", errors="coerce")
    df = _drop(df, sd.isna(), "invalid_signup_date", errs, t)
    age = pd.to_numeric(df.age, errors="coerce")
    df = _drop(df, df.age.notna() & age.isna(), "invalid_type_age", errs, t)
    age = pd.to_numeric(df.age, errors="coerce")
    df = _drop(df, age.notna() & ((age < 13) | (age > 100)), "age_out_of_range", errs, t)
    return df.assign(age=pd.to_numeric(df.age, errors="coerce"),
                     signup_date=pd.to_datetime(df.signup_date, format="%Y-%m-%d"))


def validate_items(df, errs):
    t = "items"
    df = _drop(df, df.item_id.isna(), "missing_item_id", errs, t)
    df = _drop(df, df.duplicated("item_id"), "duplicate_item_id", errs, t)
    df = _drop(df, df.category.isna(), "missing_category", errs, t)
    price = pd.to_numeric(df.price, errors="coerce")
    df = _drop(df, price.isna(), "invalid_type_price", errs, t)
    price = pd.to_numeric(df.price, errors="coerce")
    df = _drop(df, price <= 0, "non_positive_price", errs, t)
    return df.assign(price=pd.to_numeric(df.price))


def validate_interactions(df, users, items, errs):
    t = "interactions"
    for col in ("user_id", "item_id", "rating", "timestamp"):
        df = _drop(df, df[col].isna(), f"missing_{col}", errs, t)
    r = pd.to_numeric(df.rating, errors="coerce")
    df = _drop(df, r.isna(), "invalid_type_rating", errs, t)
    r = pd.to_numeric(df.rating)
    df = _drop(df, ~r.between(1, 5), "rating_out_of_range", errs, t)
    ts = pd.to_datetime(df.timestamp, errors="coerce")
    df = _drop(df, ts.isna(), "invalid_timestamp_format", errs, t)
    ts = pd.to_datetime(df.timestamp)
    df = _drop(df, (ts > C.REFERENCE_DATE) | (ts < pd.Timestamp("2015-01-01")), "timestamp_out_of_range", errs, t)
    df = _drop(df, df.duplicated(["user_id", "item_id", "timestamp", "rating"]), "duplicate_interaction", errs, t)
    df = _drop(df, ~df.user_id.isin(users.user_id), "orphan_user_id", errs, t)
    df = _drop(df, ~df.item_id.isin(items.item_id), "orphan_item_id", errs, t)
    return df.assign(rating=pd.to_numeric(df.rating), timestamp=pd.to_datetime(df.timestamp))


def run():
    C.ensure_dirs()
    raw = {n: pd.read_csv(C.RAW / f"{n}.csv", dtype=str) for n in ("users", "items", "interactions")}
    errs = []
    users = validate_users(raw["users"], errs)
    items = validate_items(raw["items"], errs)
    inter = validate_interactions(raw["interactions"], users, items, errs)
    rejected = pd.concat(errs, ignore_index=True) if errs else pd.DataFrame(columns=["source_table", "error_reason", "record"])
    C.save_csv(rejected, C.ERR / "rejected_records.csv")
    for n, d in (("users", users), ("items", items), ("interactions", inter)):
        C.save_csv(d, C.VALID / f"{n}.csv")

    report = {"warnings": {"users_missing_age": int(users.age.isna().sum())}, "tables": {}}
    for n, d in (("users", users), ("items", items), ("interactions", inter)):
        rej = rejected[rejected.source_table == n]
        report["tables"][n] = {"input_rows": len(raw[n]), "valid_rows": len(d), "rejected_rows": len(rej),
                               "reject_rate": round(len(rej) / len(raw[n]), 4),
                               "reasons": rej.error_reason.value_counts().to_dict()}
    (C.REP / "validation_report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report)
    for n, s in report["tables"].items():
        log.info("validate %-12s in=%d ok=%d rejected=%d %s", n, s["input_rows"], s["valid_rows"], s["rejected_rows"], s["reasons"])
        if s["reject_rate"] > C.MAX_REJECT_RATE:
            raise ValueError(f"Quality gate failed for {n}: reject rate {s['reject_rate']:.1%}")


def _write_markdown(report):
    """Human-readable validation report (reports/validation_report.md)."""
    lines = ["# Data validation report", "",
             "| Table | Input rows | Valid rows | Rejected | Reject rate |", "|---|---:|---:|---:|---:|"]
    for n, s in report["tables"].items():
        lines.append(f"| {n} | {s['input_rows']} | {s['valid_rows']} | {s['rejected_rows']} | {s['reject_rate']:.2%} |")
    lines += ["", "## Rejections by reason", "", "| Table | Reason | Rows |", "|---|---|---:|"]
    for n, s in report["tables"].items():
        for reason, cnt in s["reasons"].items():
            lines.append(f"| {n} | {reason} | {cnt} |")
    lines += ["", f"Warnings (kept, handled in preparation): {report['warnings']}",
              f"Quality gate: pipeline fails if any table reject rate > {C.MAX_REJECT_RATE:.0%}.",
              "Rejected records with reasons: `data/errors/rejected_records.csv`. Note: orphan_item_id / orphan_user_id "
              "include interactions whose parent row was itself rejected (cascade)."]
    (C.REP / "validation_report.md").write_text("\n".join(lines) + "\n")
