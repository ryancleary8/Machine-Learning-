# ==== 0) CLI / Imports ====
import argparse
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupKFold,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from scipy.stats import randint, uniform

RANDOM_STATE = 42


def build_parser():
    p = argparse.ArgumentParser(
        description="California House Price Predictor (Pro features: tuning, stacking, intervals)"
    )
    p.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV to tune the HistGradientBoostingRegressor.")
    p.add_argument("--n_iter", type=int, default=15, help="Number of RandomizedSearch iterations when --tune is set.")
    p.add_argument("--cv", type=int, default=5, help="CV folds for tuning and reporting.")
    p.add_argument(
        "--stack",
        action="store_true",
        help="Use a stacked model (HGB + RandomForest) with HGB as final blender.",
    )
    p.add_argument(
        "--intervals",
        action="store_true",
        help="Also train quantile models (P10/P90) to print 80% prediction intervals.",
    )
    p.add_argument(
        "--loss",
        choices=["squared_error", "absolute_error", "quantile"],
        default="squared_error",
        help="Base loss for HGB (squared_error≈RMSE, absolute_error≈MAE, quantile for pinball).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Quantile alpha when --loss quantile (e.g., 0.5=median).",
    )
    return p


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Small, high-value feature engineering."""
    X = df.drop(columns=["MedHouseVal"]).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        X["LatLon"] = X["Latitude"] * X["Longitude"]
        X["RoomsPerHouse"] = X["AveRooms"] / X["AveOccup"]
        X["BedsPerRoom"] = X["AveBedrms"] / X["AveRooms"]
    # Mild non-linear/interaction terms
    X["Latitude2"] = X["Latitude"] ** 2
    X["Longitude2"] = X["Longitude"] ** 2
    X["MedInc2"] = X["MedInc"] ** 2
    X["MedInc_x_Lat"] = X["MedInc"] * X["Latitude"]
    X["MedInc_x_Lon"] = X["MedInc"] * X["Longitude"]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X


def base_hgb(loss: str, alpha: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss,
        quantile=alpha if loss == "quantile" else None,
        learning_rate=0.07,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        max_depth=None,
        max_bins=255,                 # finer split resolution
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )


def build_base_pipe(numeric_features, loss: str, alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ])
    hgb = base_hgb(loss, alpha)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=hgb,
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])
    return pipe


def build_stacked_pipe(numeric_features, loss: str, alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
    ])
    # Base learners kept tree-based for simplicity (no scaling required)
    hgb_base = base_hgb(loss, alpha)
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=22,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    stack = StackingRegressor(
        estimators=[("rf", rf)],
        final_estimator=hgb_base,
        passthrough=True,
        n_jobs=-1,
    )
    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", TransformedTargetRegressor(
            regressor=stack,
            func=np.log1p,
            inverse_func=np.expm1,
        )),
    ])
    return pipe


def main():
    args = build_parser().parse_args()

    # ==== 1) Load data ====
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    # ==== 2) Feature engineering ====
    X = make_features(df)
    y = df["MedHouseVal"]

    # ==== 3) Split ====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # ==== 4) Build model pipeline ====
    numeric_features = list(X.columns)
    if args.stack:
        base_pipe = build_stacked_pipe(numeric_features, args.loss, args.alpha)
    else:
        base_pipe = build_base_pipe(numeric_features, args.loss, args.alpha)

    # ==== 5) Optional hyperparameter tuning (on top of strong defaults) ====
    pipe = base_pipe
    if args.tune:
        # Target the regressor *inside* TransformedTargetRegressor
        # Use conservative ranges to keep runtime reasonable
        param_dist = {
            "model__regressor__learning_rate": uniform(0.03, 0.12),
            "model__regressor__max_leaf_nodes": randint(16, 256),
            "model__regressor__min_samples_leaf": randint(5, 80),
            "model__regressor__max_bins": randint(128, 255),
        }
        # If stacking, tune final_estimator's HGB via its prefix
        if args.stack:
            param_dist = {
                "model__regressor__final_estimator__learning_rate": uniform(0.03, 0.12),
                "model__regressor__final_estimator__max_leaf_nodes": randint(16, 256),
                "model__regressor__final_estimator__min_samples_leaf": randint(5, 80),
                "model__regressor__final_estimator__max_bins": randint(128, 255),
            }
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=args.n_iter,
            scoring="r2",
            cv=args.cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print("\n[Hyperparameter Tuning] Best params:", search.best_params_)
    else:
        pipe.fit(X_train, y_train)

    # ==== 6) Evaluate (Test) ====
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)
    except TypeError:
        rmse = mean_squared_error(y_test, preds) ** 0.5

    print("\n== California House Price Predictor (Pro) ==")
    print(f"Test R^2:  {r2:.4f}   (explained variance; higher is better)")
    print(f"Test MAE:  {mae:.4f}   (~${mae*100_000:,.0f} average error)")
    print(f"Test RMSE: {rmse:.4f}   (~${rmse*100_000:,.0f} root-mean-squared error)")

    # ==== 7) CV on Train ====
    cv = KFold(n_splits=args.cv, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    print(f"\nCV R^2 ({args.cv}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Geo-aware CV
    lat_bin = (X_train["Latitude"] // 1).astype(int)
    lon_bin = (X_train["Longitude"] // 1).astype(int)
    groups = lat_bin * 1000 + lon_bin
    gkf = GroupKFold(n_splits=args.cv)
    geo_scores = cross_val_score(
        pipe, X_train, y_train, cv=gkf.split(X_train, y_train, groups), scoring="r2", n_jobs=-1
    )
    print(f"Geo-Grouped CV R^2: {geo_scores.mean():.4f} ± {geo_scores.std():.4f}")

    # ==== 8) Optional prediction intervals via quantile models ====
    if args.intervals:
        # Train two additional quantile models for 10th and 90th percentiles
        q10 = build_base_pipe(numeric_features, loss="quantile", alpha=0.1)
        q90 = build_base_pipe(numeric_features, loss="quantile", alpha=0.9)
        q10.fit(X_train, y_train)
        q90.fit(X_train, y_train)
        lo = q10.predict(X_test)
        hi = q90.predict(X_test)
        # Show intervals for the first 10 rows
        intervals_preview = pd.DataFrame({
            "Pred (x$100k)": np.round(preds[:10], 3),
            "P10": np.round(lo[:10], 3),
            "P90": np.round(hi[:10], 3),
            "Actual": np.round(y_test.iloc[:10].values, 3),
        })
        print("\nFirst 10 predictions with 80% intervals:")
        print(intervals_preview.to_string(index=False))

    # ==== 9) Preview predictions ====
    preview = pd.DataFrame({
        "Predicted (x$100k)": np.round(preds[:10], 3),
        "Actual (x$100k)": np.round(y_test.iloc[:10].values, 3),
    })
    print("\nFirst 10 predictions vs actuals:")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
