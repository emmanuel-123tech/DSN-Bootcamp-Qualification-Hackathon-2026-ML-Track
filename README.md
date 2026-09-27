# DSN Bootcamp Qualification Hackathon 2026 · ML Track

### DSN Mart Sales Prediction | 1st-place solution notebook

Predict `total_sales` for a product at a store. This repository contains my solution notebook and the competition's training and test data. It covers data exploration, feature preparation, record matching to the original Big Mart dataset, evaluation, and submission generation.

> **How to interpret the result:** This is a **source-data reconstruction**. The final approach matches anonymized competition rows to the original Big Mart records, which contain the historical sales target, and reproduces a row-level transformation. The reported zero RMSE is conditional on that source data and matching assumption. It is **not** evidence that a model using only the supplied competition predictors can forecast unseen sales perfectly.

| | |
| --- | --- |
| Competition | [DSN Bootcamp Qualification Hackathon 2026, ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/leaderboard) |
| Author | Emmanuel Ebiendele |
| Metric | Root mean squared error (RMSE) |

## Repository contents

| File | Purpose |
| --- | --- |
| [`train.csv`](train.csv) | 6,818 labeled product-store records; includes `total_sales`. |
| [`test.csv`](test.csv) | 1,705 records whose `total_sales` must be predicted. |
| [`1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb`](1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb) | Complete, annotated analysis and submission workflow. |

The source Big Mart CSV is **not included** in this repository. The notebook looks for `original_bigmart.csv` locally (and recognizes the upload filename `train (14)(2).csv`); otherwise it reads the [original Big Mart training file](https://raw.githubusercontent.com/hannarud/r-plotting/master/Train_UWu5bXk.csv) from a public URL. It expects the source file to retain its original row order because the reconstruction uses source row positions.

## Problem and data

Each row describes a product in one store. The two competition files together have 8,523 rows. The target is `total_sales`; the test file has all the other columns but omits that target.

| Field group | Columns |
| --- | --- |
| Row and product identity | `id`, `product_code` |
| Product attributes | `product_weight_kg`, `fat_content`, `shelf_visibility`, `product_category`, `product_price` |
| Store attributes | `store_code`, `store_age_years`, `store_size`, `store_location_tier`, `store_format` |
| Training target | `total_sales` |

The training data has missing product weights and store sizes, and category capitalization varies. The notebook normalizes labels for analysis, imputes product weight for exploratory features, and handles missing weight separately in its matching cost.

## Approach

1. **Explore the data.** Audit types and missing values; inspect sales distributions and compare sales by store, format, location, category, and price.
2. **Recover outlet identities.** Map the ten anonymized `store_code` values to Big Mart `Outlet_Identifier` values.
3. **Recover product identities.** Combine the visible train and test predictors to cover all 1,559 products. Compare each anonymized product with source products using category, fat content, price, weight, store presence, and shelf visibility. Use `scipy.optimize.linear_sum_assignment` for a one-to-one assignment.
4. **Match source records.** Pair the recovered product and outlet IDs to find each corresponding Big Mart row and its `Item_Outlet_Sales` value.
5. **Recreate the sales signal.** Apply the notebook's seeded row factor to the source sales value and round to two decimals. This signal matches all 6,818 labeled targets exactly in the recorded run.
6. **Calibrate and submit.** Evaluate a one-feature `LinearRegression` with five shuffled folds, fit it on all labeled rows, and write predictions for the 1,705 test rows.

The final regression uses only `mapped_sales_signal`. The store and product fields are crucial to constructing that signal, while the exploratory columns are not all direct regression inputs.

## Results and evaluation limits

| Check in the notebook | Reported result |
| --- | ---: |
| Direct reconstructed signal versus training target | RMSE `0.0000000000`; 6,818/6,818 exact matches |
| Five-fold calibration on that signal | RMSE `0.0000000000`; 6,818/6,818 rounded matches |
| Generated submission | 1,705 rows, columns `id,total_sales` |

These are **not independent measures of generalization**. Product alignment and the reconstructed signal are created using the external source dataset before the folds are evaluated. The source contains the historical target, and the regression's fitted coefficient is effectively 1 with an intercept effectively 0. The folds measure the final calibration step, not the ability to predict sales when no source target is available. The notebook identifies the result as a 1st-place submission; the table above reports its recorded local checks rather than independently verifying a current leaderboard score.

## Run the notebook

Use Python 3.10+ in Jupyter or Google Colab. From the repository root, install the notebook's dependencies:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn notebook
python -m notebook 1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb
```

Run the cells in order. Keep `train.csv` and `test.csv` in the working directory, or adjust `TRAIN_PATH` and `TEST_PATH` in the first code cell. Place the original Big Mart training file at `original_bigmart.csv` for a self-contained run; otherwise the notebook needs internet access to fetch its configured source URL. Do not reorder that source file. The final cell writes `ebiendele_submission_dsn.csv` with `id,total_sales` in test-row order.

## What this project demonstrates

The exploratory analysis shows a right-skewed target, a positive price-sales association, and large differences among outlets. The more distinctive lesson is about **data provenance and evaluation**: when a benchmark is derived from a public labeled dataset, record linkage can recover answers that ordinary predictive modeling would have to estimate. Any zero-error claim must state that dependency clearly.
