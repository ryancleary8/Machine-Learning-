# ==== 0) CLI / Imports ====
import argparse
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, GroupKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.stats import randint, uniform

RANDOM_STATE = 42

def build_parser():
    p = argparse.ArgumentParser(description="California House Price Predictor (Upgraded + Optional Tuning)")
    p.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV to tune the HistGradientBoostingRegressor.")
    p.add_argument("--n_iter", type=int, default=15, help="Number of RandomizedSearch iterations when --tune is set.")
    p.add_argument("--cv", type=int, default=5, help="CV folds for tuning and reporting.")
    return p

def main():
    args = build_parser().parse_args()

    # ==== 1) Load data ====
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # features + target (MedHouseVal)

    # ==== 2) Feature engineering (small, high-value) ====
    X = df.drop(columns=["MedHouseVal"]).copy()

    # Base engineered features
    with np.errstate(divide="ignore", invalid="ignore"):
        X["LatLon"] = X["Latitude"] * X["Longitude"]
        X["RoomsPerHouse"] = X["AveRooms"] / X["AveOccup"]
        X["BedsPerRoom"] = X["AveBedrms"] / X["AveRooms"]

    # Extra mild non-linear/interaction terms (help with geo + income curvature)
    X["Latitude2"] = X["Latitude"] ** 2
    X["Longitude2"] = X["Longitude"] ** 2
    X["MedInc2"] = X["MedInc"] ** 2
    X["MedInc_x_Lat"] = X["MedInc"] * X["Latitude"]
    X["MedInc_x_Lon"] = X["MedInc"] * X["Longitude"]

    # Clean up infinities from divisions
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Target in $100,000s
    y = df["MedHouseVal"]

    # ==== 3) Split ====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ==== 4) Preprocessing (numeric impute only; trees don't need scaling) ====
    numeric_features = list(X.columns)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
    ])

    # ==== 5) Base model ====
    hgb = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        max_depth=None,
        random_state=RANDOM_STATE
    )
    base_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=hgb,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    # ==== 6) Optional hyperparameter tuning ====
    if args.tune:
        # Parameter space for the *regressor* inside TransformedTargetRegressor
        param_dist = {
            "model__regressor__learning_rate": uniform(0.03, 0.12),   # ~0.03–0.15
            "model__regressor__max_leaf_nodes": randint(16, 256),
            "model__regressor__min_samples_leaf": randint(5, 80),
            "model__regressor__max_bins": randint(64, 255)            # improves splits on continuous vars
        }
        search = RandomizedSearchCV(
            base_pipe,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="r2",
            cv=args.cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print("\n[Hyperparameter Tuning] Best params:", search.best_params_)
    else:
        pipe = base_pipe
        pipe.fit(X_train, y_train)

    # ==== 7) Evaluate (Test) ====
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_test, preds) ** 0.5

    print("\n== California House Price Predictor (Improved) ==")
    print(f"Test R^2:  {r2:.4f}   (explained variance; higher is better)")
    print(f"Test MAE:  {mae:.4f}   (~${mae*100_000:,.0f} average error)")
    print(f"Test RMSE: {rmse:.4f}   (~${rmse*100_000:,.0f} root-mean-squared error)")

    # ==== 8) CV on Train ====
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\nCV R^2 ({args.cv}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Geo-aware CV
    lat_bin = (X_train["Latitude"] // 1).astype(int)
    lon_bin = (X_train["Longitude"] // 1).astype(int)
    groups = lat_bin * 1000 + lon_bin
    gkf = GroupKFold(n_splits=args.cv)
    geo_scores = cross_val_score(pipe, X_train, y_train, cv=gkf.split(X_train, y_train, groups), scoring="r2", n_jobs=-1)
    print(f"Geo-Grouped CV R^2: {geo_scores.mean():.4f} ± {geo_scores.std():.4f}")

    # ==== 9) Preview ====
    preview = pd.DataFrame({
        "Predicted (x$100k)": np.round(preds[:10], 3),
        "Actual (x$100k)":    np.round(y_test.iloc[:10].values, 3),
    })
    print("\nFirst 10 predictions vs actuals:")
    print(preview.to_string(index=False))

if __name__ == "__main__":
    main()
