# 🏠 California Housing Price Predictor

Welcome to my project where I walk through a **progressive machine learning pipeline** for predicting California house prices using the **California Housing dataset** from scikit-learn.  
Each test file represents a new stage in my journey — starting from a basic linear regression model and advancing to a fully featured boosted ensemble with tuning, stacking, and interval prediction.

---

## ⚙️ Setup

To get started, here’s how I set up the environment:

```bash
# Create and activate virtual environment
python -m venv env
source env/bin/activate  # macOS/Linux
# .\env\Scripts\activate  # Windows

# Install dependencies
pip install scikit-learn numpy pandas joblib scipy
```

---

## 🧩 Test 1 — Simple Linear Regression

**File:** `Test1.py`

For my first test, I kept things simple:

- I used a **LinearRegression** model.
- Applied standard preprocessing — imputing missing values and scaling features.
- Predicted the median house value (`MedHouseVal`) directly.

This was my baseline model — straightforward and interpretable.

**Key Metrics I observed (Typical):**
- R² around 0.57  
- MAE roughly 0.53 (about $53,000)  
- RMSE near 0.75 (about $75,000)

If you want to try it out:

```bash
python Test1.py
```

---

## 🚀 Test 2 — Gradient Boosted Trees (HistGradientBoosting)

**File:** `Test2.py`

Next, I decided to upgrade the model:

- I replaced Linear Regression with **HistGradientBoostingRegressor**.
- Added a **log-transform** on the target using `TransformedTargetRegressor` to stabilize variance and improve performance.
- Engineered new features like:
  - Latitude × Longitude (`LatLon`)
  - Rooms per household (`RoomsPerHouse`)
  - Beds per room (`BedsPerRoom`)
- Incorporated **cross-validation (CV)** and even geo-aware CV to better evaluate the model.

With these changes, I saw typical results like:
- R² between 0.82 and 0.84  
- MAE around 0.31 (about $31,000)  
- RMSE close to 0.47 (about $47,000)

To run this:

```bash
python Test2.py
```

---

## 🧠 Test 3 — Tunable Model with Nonlinear Features

**File:** `Test3..py`

Building on Test 2, I wanted to get more sophisticated:

- Added nonlinear and interaction features such as:
  - Latitude², Longitude², MedInc²
  - MedInc × Latitude, MedInc × Longitude
- Introduced optional hyperparameter tuning with `RandomizedSearchCV`.
- Made CV folds (`--cv`) and tuning iterations (`--n_iter`) configurable.
- Improved HistGradientBoosting parameters like learning rate, depth, and max bins.

You can run the baseline like this:

```bash
python Test3..py
```

Or with tuning enabled (20 iterations):

```bash
python Test3..py --tune --n_iter 20
```

I noticed a slight R² boost of about 1–2%, and overall better generalization.

---

## 🧩 Test 4 — Pro Model (Stacking, Tuning, Quantile Intervals)

**File:** `Test4.py`

This was my most advanced step:

- Added an optional **stacked model** (`--stack`) combining Random Forest and HistGradientBoosting.
- Kept **optional tuning** (`--tune`, `--n_iter`).
- Introduced **quantile loss** and control over alpha (`--loss`, `--alpha`).
- Enabled **prediction intervals** through quantile models (`--intervals`).
- Improved defaults like max bins, early stopping, and validation fraction.
- Supported multiple loss types:
  - `squared_error` for RMSE
  - `absolute_error` for MAE
  - `quantile` for percentile modeling

### Some example runs I tried:

#### Basic (no tuning)
```bash
python Test4.py
```

#### Tune best parameters
```bash
python Test4.py --tune --n_iter 20
```

#### Add stacking for ensemble learning
```bash
python Test4.py --stack
```

#### Show 80% prediction intervals
```bash
python Test4.py --intervals
```

#### Combine everything (heaviest run)
```bash
python Test4.py --stack --tune --n_iter 20 --intervals
```

This step took about 5–10 minutes on my laptop. The R² I achieved was around 0.83–0.86 with tuning and stacking, and the average error was roughly $28K–$32K.

---

## 🧮 Summary of Model Progression

| Test | Model Type | Key Additions | Typical R² | Avg. Error (MAE) | Notes |
|------|-------------|----------------|-------------|------------------|-------|
| 1 | Linear Regression | Baseline numeric preprocessing | ~0.57 | $53k | Simple but weak |
| 2 | HistGradientBoosting | Log target, tree model, engineered ratios | ~0.83 | $31k | Major improvement |
| 3 | Tunable HGB | Nonlinear & interaction features, CV tuning | ~0.84 | $30k | More robust generalization |
| 4 | Stacked Ensemble | RandomForest + HGB, tuning, quantile intervals | ~0.85+ | $28k–$32k | Most advanced; supports intervals and stacking |

---

## 💡 Future Improvements

Looking ahead, I’m considering:

- Saving and loading trained models with `joblib`.
- Adding progress bars and runtime timers for better feedback.
- Experimenting with other powerful frameworks like **XGBoost**, **LightGBM**, or **CatBoost**.
- Deploying the model through a **FastAPI** or **Streamlit** web interface for easier access.

Thanks for checking out my project!
