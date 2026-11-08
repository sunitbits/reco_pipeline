# Data validation report

| Table | Input rows | Valid rows | Rejected | Reject rate |
|---|---:|---:|---:|---:|
| users | 1505 | 1493 | 12 | 0.80% |
| items | 603 | 591 | 12 | 1.99% |
| interactions | 58715 | 54212 | 4503 | 7.67% |

## Rejections by reason

| Table | Reason | Rows |
|---|---|---:|
| users | duplicate_user_id | 5 |
| users | age_out_of_range | 4 |
| users | invalid_signup_date | 3 |
| items | missing_category | 4 |
| items | duplicate_item_id | 3 |
| items | non_positive_price | 3 |
| items | invalid_type_price | 2 |
| interactions | orphan_item_id | 1706 |
| interactions | duplicate_interaction | 562 |
| interactions | orphan_user_id | 470 |
| interactions | missing_rating | 295 |
| interactions | rating_out_of_range | 295 |
| interactions | invalid_timestamp_format | 295 |
| interactions | timestamp_out_of_range | 295 |
| interactions | invalid_type_rating | 293 |
| interactions | missing_user_id | 292 |

Warnings (kept, handled in preparation): {'users_missing_age': 8}
Quality gate: pipeline fails if any table reject rate > 10%.
Rejected records with reasons: `data/errors/rejected_records.csv`. Note: orphan_item_id / orphan_user_id include interactions whose parent row was itself rejected (cascade).
