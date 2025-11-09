# ==== 1) Imports ====
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge

RANDOM_STATE = 42

# ==== 2) Load data ====
# as_frame=True gives a pandas DataFrame directly
data = fetch_california_housing(as_frame=True)
df = data.frame  # all columns + target
X = df.drop(columns=["MedHouseVal"])  # features
y = df["MedHouseVal"]                 # target (in $100,000s)

# ==== 3) Train/Test split ====
# Simple 80/20 split; shuffle for randomness
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ==== 4) Preprocessing ====
# All features here are numeric; we:
#  - fill any missing values with median
#  - scale features to mean=0, std=1 (helps linear regression)
numeric_features = list(X.columns)
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
])

# ==== 5) Model ====
model = LinearRegression()

# ==== 6) Full pipeline (preprocess -> model) ====
linear = Ridge(alpha=1.0, random_state=RANDOM_STATE)
linear = LinearRegression()
pipe = Pipeline([
    ("prep", preprocessor),
    ("model", TransformedTargetRegressor(regressor=linear,
                                         func=np.log1p,      # y -> log(1+y)
                                         inverse_func=np.expm1))  # back to original units
])

# ==== 7) Train ====
pipe.fit(X_train, y_train)

# ==== 8) Evaluate ====
preds = pipe.predict(X_test)

r2 = r2_score(y_test, preds)                          # higher is better (<=1.0)
mae = mean_absolute_error(y_test, preds)              # avg absolute error
try:
    rmse = mean_squared_error(y_test, preds, squared=False)
except TypeError:
    rmse = mean_squared_error(y_test, preds) ** 0.5

print("\n== California House Price Predictor (Simple) ==")
print(f"R^2:  {r2:.4f}   (explained variance; higher is better)")
print(f"MAE:  {mae:.4f}   (~${mae*100_000:,.0f} average error)")
print(f"RMSE: {rmse:.4f}   (~${rmse*100_000:,.0f} root-mean-squared error)")

# ==== 9) Peek at predictions ====
preview = pd.DataFrame({
    "Predicted (x$100k)": np.round(preds[:10], 3),
    "Actual (x$100k)":    np.round(y_test.iloc[:10].values, 3),
})
print("\nFirst 10 predictions vs actuals:")
print(preview.to_string(index=False))

