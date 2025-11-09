# ==== 1) Imports ====
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

RANDOM_STATE = 42

# ==== 2) Load data ====
# as_frame=True gives a pandas DataFrame directly
data = fetch_california_housing(as_frame=True)
df = data.frame  # all columns + target

# ==== 3) Feature engineering (small, high-value) ====
# Start from features (drop target)
X = df.drop(columns=["MedHouseVal"]).copy()

# Add simple interactions/ratios that help tree models
# - Latitude * Longitude captures coastal/inland gradient
# - Rooms per household and Beds per room add density/quality signals
with np.errstate(divide="ignore", invalid="ignore"):
    X["LatLon"] = X["Latitude"] * X["Longitude"]
    X["RoomsPerHouse"] = X["AveRooms"] / X["AveOccup"]
    X["BedsPerRoom"] = X["AveBedrms"] / X["AveRooms"]

# Replace inf/-inf from any divide-by-zero with NaN so imputer can handle them
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Target (in $100,000s)
y = df["MedHouseVal"]

# ==== 4) Train/Test split ====
# Simple 80/20 split; shuffle for randomness
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ==== 5) Preprocessing ====
# All features here are numeric; for tree models we only need imputation.
numeric_features = list(X.columns)
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
])

# ==== 6) Model ====
# HistGradientBoosting is fast and strong for tabular regression.
hgb = HistGradientBoostingRegressor(
    learning_rate=0.07,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    max_depth=None,
    random_state=RANDOM_STATE
)

# Log-transform the target for stability; predictions are inverse-transformed back.
pipe = Pipeline([
    ("prep", preprocessor),
    ("model", TransformedTargetRegressor(
        regressor=hgb,
        func=np.log1p,       # y -> log(1+y)
        inverse_func=np.expm1  # back to original units
    ))
])

# ==== 7) Train ====
pipe.fit(X_train, y_train)

# ==== 8) Evaluate (Test set) ====
preds = pipe.predict(X_test)

r2 = r2_score(y_test, preds)                          # higher is better (<=1.0)
mae = mean_absolute_error(y_test, preds)              # avg absolute error
try:
    rmse = mean_squared_error(y_test, preds, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, preds) ** 0.5

print("\n== California House Price Predictor (Upgraded) ==")
print(f"Test R^2:  {r2:.4f}   (explained variance; higher is better)")
print(f"Test MAE:  {mae:.4f}   (~${mae*100_000:,.0f} average error)")
print(f"Test RMSE: {rmse:.4f}   (~${rmse*100_000:,.0f} root-mean-squared error)")

# ==== 9) Cross-validation on Train (robustness) ====
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
print(f"\nCV R^2 (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Geography-aware CV (avoid leaking neighborhood effects across folds)
lat_bin = (X_train["Latitude"] // 1).astype(int)
lon_bin = (X_train["Longitude"] // 1).astype(int)
groups = lat_bin * 1000 + lon_bin
gkf = GroupKFold(n_splits=5)
geo_scores = cross_val_score(pipe, X_train, y_train, cv=gkf, groups=groups, scoring="r2", n_jobs=-1)
print(f"Geo-Grouped CV R^2: {geo_scores.mean():.4f} ± {geo_scores.std():.4f}")

# ==== 10) Peek at predictions ====
preview = pd.DataFrame({
    "Predicted (x$100k)": np.round(preds[:10], 3),
    "Actual (x$100k)":    np.round(y_test.iloc[:10].values, 3),
})
print("\nFirst 10 predictions vs actuals:")
print(preview.to_string(index=False))
