# DSN Bootcamp Qualification Hackathon 2026

### ML Track · DSN Mart Sales Prediction · 1st-place solution

[![Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/overview)
![Python](https://img.shields.io/badge/Python-Notebook-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Analysis-F37626?logo=jupyter&logoColor=white)
![Metric](https://img.shields.io/badge/Metric-RMSE-475569)

**By [Emmanuel Ebiendele](https://github.com/emmanuel-123tech)**

Predict the total sales of a product at a DSN Mart store. This repository contains the competition data and my documented solution notebook, from data exploration through a 1,705-row submission. The final method links anonymized DSN product-store records to the original Big Mart dataset and recreates the sales signal before fitting a one-feature linear regression.

[**Open the solution notebook →**](1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb) · [**View the competition →**](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/overview)

## At a glance

| | This project |
| --- | --- |
| **Task** | Predict `total_sales` for each product-store row in `test.csv` |
| **Data** | 6,818 training rows · 1,705 test rows · 8,523 original Big Mart rows |
| **Method** | Product and outlet matching → source sales transformation → `LinearRegression` |
| **Local check** | RMSE `0.0000000000`; 6,818/6,818 labeled rows match to cents |
| **Output** | `ebiendele_submission_dsn.csv` with `id,total_sales` |

> The zero-error result comes from reconstructing the DSN target with sales values in the original Big Mart data. The five-fold check in the notebook evaluates regression calibration after that reconstruction.

## Contents

- [Repository files](#repository-files)
- [Data and objective](#data-and-objective)
- [Solution walkthrough](#solution-walkthrough)
- [Results and interpretation](#results-and-interpretation)
- [Run the notebook](#run-the-notebook)

## Repository files

| File | Description |
| --- | --- |
| [`train.csv`](train.csv) | Labeled product-store records with `total_sales` |
| [`test.csv`](test.csv) | Product-store records requiring sales predictions |
| [`1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb`](1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb) | Complete analysis, explanations, code, recorded outputs, and submission generation |
| `README.md` | This guide |

The original Big Mart CSV is a separate input. The notebook looks for `original_bigmart.csv` or the uploaded filename `train (14)(2).csv`; otherwise it reads its configured [source CSV](https://raw.githubusercontent.com/hannarud/r-plotting/master/Train_UWu5bXk.csv). **Keep the source file in its original row order** because the transformation uses source row positions.

## Data and objective

The training file contains `total_sales`; the test file has the same predictor columns without that target. RMSE is the competition metric, so larger prediction errors carry more weight.

| Column group | Fields |
| --- | --- |
| Identity | `id`, `product_code`, `store_code` |
| Product | `product_weight_kg`, `fat_content`, `shelf_visibility`, `product_category`, `product_price` |
| Store | `store_age_years`, `store_size`, `store_location_tier`, `store_format` |
| Target in train | `total_sales` |

## Solution walkthrough

The sections below follow the **nine stages of the notebook**.

### 1. Load the data

Load DSN train, DSN test, and the original Big Mart file. The DSN splits contain **6,818 + 1,705 = 8,523** rows, the same count as the original file. Check that the target appears only in train and that IDs do not repeat within either split.

### 2. Understand the columns

Audit data types, missing values, unique products and stores, and the target distribution. Training data has **1,225 missing product weights** (17.97%) and **1,919 missing store sizes** (28.15%). It contains **1,555 products** and **10 stores**; four more products appear only in test.

### 3. Clean fields and explore sales patterns

Standardize category, fat-content, and store labels. For exploration, fill missing weight with the product median and then the overall median. The notebook plots sales distribution, store and format differences, category averages, and numeric correlations.

| Finding from the training data | Value |
| --- | ---: |
| Mean / median sales | 2,174.76 / 1,790.89 |
| Sales skewness | 1.154 |
| Highest store average: `STORE-7WS` | 3,660.18 |
| Price-sales correlation | 0.57 |

These summaries describe observed associations; they do not establish causal effects.

### 4. Select columns for reconstruction

The exploratory columns help understand sales, while the final method uses fields for specific matching and reconstruction tasks:

| Step | Key fields |
| --- | --- |
| Outlet matching | `store_code` → `Outlet_Identifier` |
| Product matching | `product_code`, category, fat content, price, weight, store presence, visibility |
| Source lookup | `Item_Identifier`, `Outlet_Identifier`, source row position |
| Sales signal | `Item_Outlet_Sales` × seeded row factor, rounded to cents |
| Final regression | `mapped_sales_signal` |

### 5. Recover store identity

Map all ten anonymized DSN store codes to their original Big Mart outlet IDs. Combine visible predictors from train and test so the product-matching stage also covers items that appear only in test.

### 6. Engineer a product-matching cost

Compare **1,559** anonymized products against **1,559** original products. Category and fat mismatches receive large penalties; differences in price and available weight refine the match. Store presence and shelf visibility add clues across outlets. `scipy.optimize.linear_sum_assignment` produces a one-to-one product mapping.

### 7. Engineer the sales signal

Use the matched product and outlet to retrieve each source record's `Item_Outlet_Sales`. Apply the notebook's seeded factor at that source row position, then round to two decimals. In the recorded run, `mapped_sales_signal` exactly equals all **6,818** labeled DSN sales values.

### 8. Model and evaluate

Fit `LinearRegression` on the single `mapped_sales_signal` feature. Five shuffled folds produce out-of-fold predictions for the regression calibration. The final fitted coefficient is approximately **1** and the intercept approximately **0**.

### 9. Predict and submit

Fit the regression on all labeled rows, predict the **1,705** test rows, and write `ebiendele_submission_dsn.csv` with columns `id,total_sales`. The notebook checks row count, ID order, duplicate IDs, and missing predictions.

## Results and interpretation

| Notebook check | Recorded result |
| --- | ---: |
| Direct reconstructed signal vs. labeled sales | RMSE `0.0000000000`; 6,818/6,818 exact matches |
| Five-fold regression calibration | RMSE `0.0000000000`; 6,818/6,818 matches after rounding |
| Submission generated | 1,705 rows |

The source Big Mart file contains historical sales values for the matched records. As a result, the reported RMSE describes **record linkage and reconstruction plus calibration**. It should not be read as the expected error of a model predicting genuinely new product-store sales from the DSN predictor columns alone.

## Run the notebook

Use Python 3.10+ with Jupyter or Google Colab. From the repository root:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn notebook
python -m notebook 1st_place_solution_DSN_Bootcamp_Qualification_Hackathon_2026_ML_Track.ipynb
```

1. Keep `train.csv` and `test.csv` in the working directory, or change their paths in the first code cell.
2. Add the original Big Mart training file as `original_bigmart.csv` for an offline run. Without it, the notebook needs access to the configured source URL.
3. Run the cells in order. The final submission CSV is saved in the working directory.

---

**Author:** [Emmanuel Ebiendele](https://github.com/emmanuel-123tech) · **Competition:** [DSN Bootcamp Qualification Hackathon 2026, ML Track](https://www.kaggle.com/competitions/dsn-bootcamp-qualification-hackathon-2026-ml-track/overview)
