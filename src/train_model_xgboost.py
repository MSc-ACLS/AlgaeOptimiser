import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from preprocessing import preprocess_data

def select_numeric_columns(df, exclude=("timestring", "Dryweight"), max_nan_frac=0.3):
    num = df.select_dtypes(include=["number"]).copy()
    # keep timestring & target
    if "timestring" in df.columns:
        num["timestring"] = df["timestring"]
    if "Dryweight" in df.columns:
        num["Dryweight"] = df["Dryweight"]
    # drop columns with too many NaNs (except timestring/target)
    keep = []
    for c in num.columns:
        if c in exclude or c == "timestring":
            keep.append(c)
        else:
            frac = num[c].isna().mean()
            if frac <= max_nan_frac:
                keep.append(c)
    return num[keep]

def create_lagged_features(df, target_col="Dryweight", lags=10):
    # Ensure sorted by time
    df = df.sort_values("timestring").reset_index(drop=True)

    # Use only numeric & reasonably complete columns
    df = select_numeric_columns(df, exclude=("timestring", target_col), max_nan_frac=0.3)

    # Drop rows with missing target BEFORE lagging (use ONLY measured targets)
    df = df[df[target_col].notna()].reset_index(drop=True)
    if len(df) < lags + 5:
        # Not enough rows to build requested lags
        lags = max(1, min(5, len(df) - 2))

    to_lag = [c for c in df.columns if c not in ("timestring", target_col)]
    frames = [df[["timestring", target_col]].copy()]

    # predictors lags
    for lag in range(1, lags + 1):
        shifted = df[to_lag].shift(lag)
        shifted.columns = [f"{c}_lag{lag}" for c in to_lag]
        frames.append(shifted)
    # target lags
    for lag in range(1, lags + 1):
        frames.append(df[[target_col]].shift(lag).rename(columns={target_col: f"{target_col}_lag{lag}"}))

    out = pd.concat(frames, axis=1)

    # Optionally fill small gaps in predictors only (not target)
    pred_cols = [c for c in out.columns if c not in ("timestring", target_col) and not c.startswith(f"{target_col}_")]
    out[pred_cols] = out[pred_cols].interpolate(limit_direction="both")

    out = out.dropna().reset_index(drop=True)
    return out

def safe_time_split(n, train_frac=0.7, val_frac=0.1, min_each=50):
    # Compute indices with minimum sizes; fall back if too small
    i_train_end = int(n * train_frac)
    i_val_end   = int(n * (train_frac + val_frac))
    # Enforce minimums if possible
    if i_train_end < min_each and n >= 3 * min_each:
        i_train_end = min_each
        i_val_end = min_each * 2
    # If dataset is small, switch to 80/20 (no val)
    if n - i_val_end < min_each:
        i_train_end = int(n * 0.8)
        i_val_end = i_train_end  # empty val set: we'll reuse a piece of train for early stopping
    return i_train_end, i_val_end

def train_xgb(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.0,
        random_state=42,
        tree_method="hist",
        n_jobs=-1,
        eval_metric="rmse",
        early_stopping_rounds=100
    )
    eval_set = [(X_train.iloc[-min(200, len(X_train)):], y_train.iloc[-min(200, len(y_train)):])]
    # If we have a non-empty val set, use it for early stopping; otherwise use a tail of train
    if len(X_val) > 0:
        eval_set = [(X_val, y_val)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
    return model

def evaluate_model(model, X_test, y_test, feature_names, top_k=20):
    from xgboost import plot_importance  # optional
    if len(X_test) == 0:
        print("Test set empty; adjust split or reduce lags.")
        return
    y_pred = model.predict(X_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    booster = model.get_booster()
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
    raw_imp = booster.get_score(importance_type="gain")
    if not raw_imp:
        print("\nNo feature importance available.")
        return
    imp_df = (
        pd.Series({fmap.get(k, k): v for k, v in raw_imp.items()})
          .sort_values(ascending=False)
          .head(top_k)
          .reset_index()
    )
    imp_df.columns = ["feature", "importance_gain"]
    print("\nTop feature importances (gain):")
    print(imp_df.to_string(index=False))

def main():
    zhaw_data, _ = preprocess_data()  # ensure Dryweight is NOT globally interpolated
    data_supervised = create_lagged_features(zhaw_data, target_col="Dryweight", lags=10)

    X = data_supervised.drop(columns=["Dryweight", "timestring"])
    y = data_supervised["Dryweight"]

    n = len(X)
    print(f"Supervised rows after lagging & dropna: {n}, features: {X.shape[1]}")
    i_train_end, i_val_end = safe_time_split(n, train_frac=0.7, val_frac=0.1, min_each=50)

    X_train, y_train = X.iloc[:i_train_end], y.iloc[:i_train_end]
    X_val,   y_val   = X.iloc[i_train_end:i_val_end], y.iloc[i_train_end:i_val_end]
    X_test,  y_test  = X.iloc[i_val_end:], y.iloc[i_val_end:]

    print(f"Split sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    model = train_xgb(X_train, y_train, X_val, y_val)
    evaluate_model(model, X_test, y_test, feature_names=X.columns.tolist(), top_k=20)

if __name__ == "__main__":
    main()
