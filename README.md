# Customer Purchase Intent Prediction

Predicting whether an e-commerce session converts, from session-level behavioural
signals (time on site, pages viewed, cart value, ad engagement, referral source).
The data is imbalanced (only ~29% of sessions convert), so the project is built
around recall, F1, and PR-AUC on the purchasing class rather than raw accuracy.

End-to-end in a single notebook: `code.ipynb`.

## Headline results

- **Final model: tuned XGBoost**, selected by PR-AUC (**0.692**, top of three benchmarked models).
- **Recall on actual buyers of 0.85** at the default threshold; cost-sensitive learning lifts recall on the minority class well above the untuned baseline.
- **Decision-threshold analysis** surfaces the precision/recall trade-off explicitly (F1-optimal point at 0.61, F1 0.691).
- **Every prediction is auditable** via SHAP feature contributions.

## Results table

Three algorithm families benchmarked (linear, bagging, boosting), each tuned with
stratified 5-fold CV scored on PR-AUC; evaluated on a held-out test set.

| Model (tuned)        | PR-AUC | Recall | F1    | ROC-AUC | Accuracy |
|----------------------|:------:|:------:|:-----:|:-------:|:--------:|
| **XGBoost**          | **0.692** | 0.846 | 0.659 | 0.854 | 0.736 |
| Logistic Regression  | 0.651 | 0.831 | 0.667 | 0.834 | 0.749 |
| Random Forest        | 0.646 | 0.750 | 0.662 | 0.845 | 0.769 |

XGBoost leads on PR-AUC, recall, and ROC-AUC. It is selected as the boosting
representative; plain gradient boosting performed comparably and is omitted to
keep the comparison to one model per family.

## Highlights

- **Metric discipline:** accuracy is reported but never optimised; on a 29%-positive problem it is misleading by construction, so recall, F1, and PR-AUC on the purchasing class drive every decision.
- **PR-AUC for selection:** unlike ROC-AUC, it excludes true negatives, so it is not inflated by the easy majority class. The selection metric was fixed before evaluation.
- **No data leakage:** outlier bounds and all preprocessing (encoding, log-transform, scaling) are fit on training data only, inside a single `ColumnTransformer` pipeline.
- **Domain feature engineering:** instead of dropping `Pages_viewed` (collinear with `Time_on_site`), engineered `engagement_intensity` and `cart_per_page`; SHAP confirmed both carry real weight.
- **Threshold analysis:** swept the 0.5 cut-off on the PR curve to the F1-optimal point (0.61), making the precision/recall trade-off explicit rather than accepting a default.
- **Explainability:** SHAP (`TreeExplainer`) ranks `Clicked_ad`, `Pages_viewed`, `Time_on_site`, then `Cart_value` as the top purchase drivers.

## Pipeline

EDA → leakage-safe outlier handling → feature engineering → leakage-free
preprocessing → 3 cost-sensitive models → stratified PR-AUC tuning → model
selection → threshold analysis → SHAP.

## Stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn.

## Run

Built to run top-to-bottom on Google Colab. Upload `customer_behavior_train.csv`
and `customer_behavior_test.csv` when prompted, then run all cells. The first cell
installs the two non-default dependencies (`xgboost`, `shap`).
