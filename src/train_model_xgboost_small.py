import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import preprocess_data

# 1) Load
zhaw_data, _ = preprocess_data()  # ensure Dryweight is NOT globally interpolated

# Keep only rows with measured Dryweight; sort by time
df = zhaw_data.sort_values("timestring").loc[zhaw_data["Dryweight"].notna(), ["timestring", "Dryweight"]].copy()
n = len(df)
if n < 5:
    raise RuntimeError(f"Too few Dryweight points ({n}). You need at least ~5 to do any split.")

# 2) Choose k (window) dynamically so we don’t drop everything
# We need at least 2 test samples after lagging. Use at most n-3 lags; cap to 8.
k = min(8, max(1, n - 3))

# Build target lags
for i in range(1, k + 1):
    df[f"DW_lag{i}"] = df["Dryweight"].shift(i)

df = df.dropna().reset_index(drop=True)
m = len(df)
if m < 5:
    # Still too small after lagging: reduce k and rebuild once
    k = max(1, min(3, n - 3))
    df = zhaw_data.sort_values("timestring").loc[zhaw_data["Dryweight"].notna(), ["timestring", "Dryweight"]].copy()
    for i in range(1, k + 1):
        df[f"DW_lag{i}"] = df["Dryweight"].shift(i)
    df = df.dropna().reset_index(drop=True)
    m = len(df)

print(f"Rows after lagging: {m}, using k={k}")

# 3) Time split with at least 2 test rows
split = max(int(m * 0.7), m - 2)   # keep >=2 in test
X_cols = [f"DW_lag{i}" for i in range(1, k + 1)]
X_train, y_train = df[X_cols].iloc[:split], df["Dryweight"].iloc[:split]
X_test,  y_test  = df[X_cols].iloc[split:],  df["Dryweight"].iloc[split:]

print(f"Split sizes — train: {len(X_train)}, test: {len(X_test)}")
if len(X_test) < 2:
    # fall back: move one row from train to test to get 2 samples
    X_test  = pd.concat([X_train.iloc[[-1]], X_test], axis=0)
    y_test  = pd.concat([y_train.iloc[[-1]], y_test], axis=0)
    X_train = X_train.iloc[:-1]
    y_train = y_train.iloc[:-1]
    print(f"Adjusted splits — train: {len(X_train)}, test: {len(X_test)}")

# 4) Train tiny XGB
model = XGBRegressor(
    n_estimators=300, max_depth=3, learning_rate=0.08,
    subsample=0.9, colsample_bytree=0.9,
    tree_method="hist", random_state=42, n_jobs=-1, eval_metric="rmse"
)
model.fit(X_train, y_train)

# 5) Evaluate
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
if len(y_test) >= 2:
    print("R2 :", r2_score(y_test, y_pred))
else:
    print("R2 : (need at least 2 test samples)")
