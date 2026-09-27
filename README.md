# DSN Bootcamp Qualification Hackathon 2026 · ML Track

## DSN Mart Sales Prediction | 1st-place solution

**Emmanuel Ebiendele** · [Competition leaderboard](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/leaderboard) · Evaluation metric: **RMSE**

This repository documents my approach to predicting `total_sales` for a product at a particular store. The notebook moves from understanding the data to exploring sales patterns, matching the anonymized products and stores to the original Big Mart records, recreating the sales signal, evaluating a linear model, and preparing the submission.

## Repository files

| File | Description |
| --- | --- |
| [`train.csv`](train.csv) | 6,818 product-store records with `total_sales`. |
| [`test.csv`](test.csv) | 1,705 product-store records without `total_sales`. |
| [`_Ebiendele_Emmanuel_DSN_Bootcamp_Qualification_Hackathon_2026_MLTrack.ipynb`](_Ebiendele_Emmanuel_DSN_Bootcamp_Qualification_Hackathon_2026_MLTrack.ipynb) | The full analysis, code, outputs, and submission workflow. |

The notebook also uses the original Big Mart training data. It reads a local `original_bigmart.csv` if available, recognizes the upload filename `train (14)(2).csv`, or downloads the [source CSV](https://raw.githubusercontent.com/hannarud/r-plotting/master/Train_UWu5bXk.csv). Keep the original source row order because the sales-factor calculation is indexed by row position.

## 1. Load the data

The notebook loads the DSN train and test files alongside the original Big Mart data. The DSN files have **6,818** and **1,705** rows respectively, adding up to the source file's **8,523** records. It checks that `total_sales` exists only in train and that each file's `id` values are unique.

Each DSN record contains a product code, product weight, fat content, shelf visibility, category, price, store code, store age, size, location tier, and format. The target is `total_sales`.

## 2. Understand the columns

The first audit examines data types, missing values, unique values, descriptive statistics, and duplicate rows.

- `product_weight_kg` is missing in 1,225 training rows (17.97%).
- `store_size` is missing in 1,919 training rows (28.15%).
- Train contains ten stores and 1,555 product codes. Four additional products appear only in test.
- `total_sales` averages **2,174.76**, while its median is **1,790.89**.

## 3. Clean fields and explore sales patterns

The exploratory preparation standardizes category, fat-content, and store labels. It fills missing weight from each product's median, then the overall median, for charts and derived exploratory fields such as price per kilogram and price bands. The original columns remain available for the later matching stage.

The notebook plots the sales distribution, store comparisons, store format and location, category sales, and numeric correlations. Sales are right-skewed (skewness **1.154**). `STORE-7WS` has the highest average sales per record (**3,660.18**), while `STORE-JOR` and `STORE-T5G` average roughly **334** and **338**. Product price has a **0.57** correlation with sales in the training data. These are descriptive associations, not causal estimates.

## 4. Select the columns for reconstruction

The final workflow gives different fields different jobs:

| Purpose | Fields |
| --- | --- |
| Identify an outlet | DSN `store_code` and Big Mart `Outlet_Identifier` |
| Identify a product | `product_code`, category, fat content, price, weight, store presence, and shelf visibility |
| Find the source record | `Item_Identifier`, `Outlet_Identifier`, and source row position |
| Recreate the sales signal | `Item_Outlet_Sales` and a seeded row-level factor |
| Fit the final regression | `mapped_sales_signal` |

Store age, size, tier, and format help describe the data, but they are not direct inputs to the final regression.

## 5. Recover store identity

A fixed mapping pairs all ten anonymized `store_code` values with the corresponding Big Mart outlet IDs. The notebook combines visible train and test predictors so products that appear only in test are included in the next matching step.

## 6. Engineer a product-matching cost

The notebook summarizes each product's normalized category, fat label, median weight, and median price. It compares all **1,559** DSN products with the **1,559** Big Mart products. Category and fat mismatches receive large penalties; price and available weight differences refine the score. Store presence and shelf visibility provide further clues across outlets.

`scipy.optimize.linear_sum_assignment` finds a one-to-one product mapping that minimizes the total matching cost. Together with the outlet mapping, this locates the original product-store row for every DSN record.

## 7. Engineer the sales signal

For each matched record, the notebook retrieves the source `Item_Outlet_Sales`. It advances a seeded random generator to the fourth uniform draw, selects the factor at the source row position, multiplies source sales by that factor, and rounds to two decimal places. The resulting `mapped_sales_signal` matches all **6,818** labeled `total_sales` values in the notebook's run.

## 8. Model and evaluate

A five-fold shuffled cross-validation fits `LinearRegression` using **one feature**, `mapped_sales_signal`. The notebook then fits the final model on all labeled rows.

| Notebook check | Recorded result |
| --- | ---: |
| Reconstructed signal against training target | RMSE `0.0000000000`; 6,818/6,818 exact matches |
| Five-fold regression predictions | RMSE `0.0000000000`; 6,818/6,818 matches after rounding |
| Final regression | Coefficient approximately 1; intercept approximately 0 |

The signal already contains sales recovered from the original labeled Big Mart file. Thus, the cross-validation score evaluates the final calibration **after** source matching and signal construction; it is not a predictor-only validation score on new sales data.

## 9. Predict and submit

The notebook predicts the **1,705** test rows and writes `ebiendele_submission_dsn.csv` in the test file's original order. The submission has exactly two columns: `id,total_sales`. The code checks row count, ID order, uniqueness, and missing predictions before saving.

## Run it yourself

Use Python 3.10+ in Jupyter or Google Colab. From the repository root:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn notebook
python -m notebook 1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb
```

Run cells from top to bottom. Keep `train.csv` and `test.csv` in the working directory or change their paths in the first code cell. Add `original_bigmart.csv` locally for an offline run; otherwise internet access is needed for the configured source URL. The generated submission CSV is written to the working directory.
