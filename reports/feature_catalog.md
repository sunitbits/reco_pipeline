# Feature catalog

All features are computed from TRAIN data only, as of the last training timestamp. Cold-start users/items get defaults from the registry.

| Feature | Entity | Type | Description |
|---|---|---|---|
| `user_activity_count` | user | int | Number of training interactions |
| `user_avg_rating` | user | float | Mean rating given by the user |
| `user_rating_std` | user | float | Std-dev of the user's ratings (0 if single rating) |
| `user_positive_ratio` | user | float | Share of the user's ratings >= 4 |
| `preferred_category` | user | str | Category with most interactions |
| `days_since_last_interaction` | user | int | Days from user's last interaction to ref_time |
| `user_tenure_days` | user | int | Days from signup to ref_time |
| `item_popularity` | item | int | Number of training interactions |
| `item_positive_count` | item | int | Number of ratings >= 4 |
| `item_popularity_pct` | item | float | Percentile rank of popularity (1 = most popular) |
| `item_avg_rating` | item | float | Bayesian-smoothed mean rating (m=10) toward global mean |
| `item_days_since_last_interaction` | item | int | Days since the item was last interacted with |
| `category / brand / price` | item | str/float | Item content attributes |
| `user_category_avg_rating` | user x category | float | User's mean rating within a category |
| `user_category_count` | user x category | int | User's interactions within a category |
| `user_category_affinity (derived)` | interaction | float | user_category_avg_rating for the item's category, else user_avg_rating |
| `rating_gap (derived)` | interaction | float | item_avg_rating - user_avg_rating |
| `is_preferred_category (derived)` | interaction | int | 1 if item category == user's preferred_category |
