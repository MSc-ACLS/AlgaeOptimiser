# file: train_mts_regression.py
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import preprocess_data

def make_supervised(df, target="Dryweight", exog_cols=None, lags_y=6, lags_exog=6):
    """ARX-style supervised table: y[t] ~ y[t-1..lags_y] + exog[t-1..lags_exog]."""
    df = df.sort_values("timestring").copy()
    # keep only columns we need and rows with measured target
    keep = ["timestring", target] + (exog_cols or [])
    df = df[[c for c in keep if c in df.columns]]
    df = df[df[target].notna()].reset_index(drop=True)

    frames = [df[["timestring", target]].copy()]

    # lag target
    for l in range(1, lags_y + 1):
        frames.append(df[[target]].shift(l).rename(columns={target: f"{target}_lag{l}"}))

    # lag exogenous
    if exog_cols:
        for l in range(1, lags_exog + 1):
            shifted = df[exog_cols].shift(l)
            shifted.columns = [f"{c}_lag{l}" for c in exog_cols]
            frames.append(shifted)

    out = pd.concat(frames, axis=1).dropna().reset_index(drop=True)

    # build X, y
    y = out[target].astype("float64")
    X = out.drop(columns=["timestring", target])
    # keep only numeric predictors
    X = X.select_dtypes(include=[np.number])

    return out, X, y

def time_split(X, y, train_frac=0.8):
    n = len(X)
    split = max(int(n * train_frac), 2)  # ensure at least 2 samples for test metrics
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]

def main():
    # Load your data (ensure NO global interpolation of Dryweight in preprocess)
    zhaw, _ = preprocess_data()

    # Choose a small, meaningful set of exogenous drivers (only those that exist)
    candidate_exog = ["PAR.1", "TEMPERATURE", "pH", "THICKNESS.OF.ALGAE",
                      "FLOW.OF.CO2", "FLOW.OF.ALGAE", "PRESSURE", "SPEED"]
    exog = [c for c in candidate_exog if c in zhaw.columns]

    # Build supervised matrix with modest lags (shrink automatically if needed)
    lags_y = 6
    lags_exog = 6
    out, X, y = make_supervised(zhaw, target="Dryweight", exog_cols=exog,
                                lags_y=lags_y, lags_exog=lags_exog)

    if len(X) < 10:
        # Fallback: reduce lags if data is sparse
        lags_y, lags_exog = 3, 3
        out, X, y = make_supervised(zhaw, target="Dryweight", exog_cols=exog,
                                    lags_y=lags_y, lags_exog=lags_exog)

    print(f"Rows after lagging: {len(X)}, features: {X.shape[1]} (lags_y={lags_y}, lags_exog={lags_exog})")

    # Time-based split
    X_train, X_test, y_train, y_test = time_split(X, y, train_frac=0.8)
    print(f"Split — train: {len(X_train)}, test: {len(X_test)}")

    # Multiple regression (standardized) with built-in CV over alpha to handle collinearity
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=np.logspace(-4, 3, 20), cv=5))
    ])
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, y_pred))
    if len(y_test) >= 2:
        print("R2 :", r2_score(y_test, y_pred))
    else:
        print("R2 : need ≥2 test samples")

    # Optional: show top coefficients by absolute value
    coef = model.named_steps["ridge"].coef_
    top = pd.Series(coef, index=X.columns).abs().sort_values(ascending=False).head(15)
    print("\nTop predictors (|coef|):")
    print(top)

if __name__ == "__main__":
    main()
