# pip install pygam pandas numpy

import pandas as pd
import numpy as np
from pygam import LinearGAM, s, te
from dataclasses import dataclass

# ---------- CONFIG ----------
ZHAW_VIRIDIELLA_DATA_CSV = "../data/zhaw_viridiella.csv"        # sensors (minute-level)
ZHAW_VIRIDIELLA_META_CSV = "../data/zhaw_viridiella_meta.csv"   # manual (DW etc.)
ZHAW_CHLAMYDOMONAS_DATA_CSV = "../data/zhaw_chlamydomonas.csv"        # sensors (minute-level)
ZHAW_CHLAMYDOMONAS_META_CSV = "../data/zhaw_chlamydomonas_meta.csv"   # manual (DW etc.)
DW_COL_META   = "Trockenmasse"                    # g/L
TIME_COL      = "timestring"

# drivers to aggregate over each interval (use what exists in your file)
SENSOR_COLS = [
    "PAR.1","PAR.2","TEMPERATURE","pH","pCO2",
    "FLOW.OF.CO2","FLOW.OF.ALGAE","SPEED","PRESSURE",
    "THICKNESS.OF.ALGAE"  # state variable (self-shading)
]

# rolling/interval window stats to compute (per interval)
AGG_FUNCS = { "mean": np.mean, "median": np.median, "min": np.min, "max": np.max }
# keep it light to start: mean only
AGG_FUNCS = { "mean": np.mean }

# ---------- IO & PREP ----------
def read_sensor_zhaw(path: str) -> pd.DataFrame:
    # skip the 2nd row with units
    df = pd.read_csv(path, skiprows=[1])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, format="%d.%m.%Y %H:%M:%S.%f")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

def read_meta_zhaw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=[1])
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], dayfirst=True, format="%d.%m.%Y %H:%M:%S.%f")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    # keep only timestamp + DW (rename for clarity)
    keep = [TIME_COL, DW_COL_META]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].rename(columns={DW_COL_META: "Dryweight"})
    # numeric coercion
    df["Dryweight"] = pd.to_numeric(df["Dryweight"], errors="coerce")
    df = df[df["Dryweight"].notna()].reset_index(drop=True)
    return df

@dataclass
class IntervalRow:
    t0: pd.Timestamp
    t1: pd.Timestamp
    dt_h: float
    d_dw: float
    g: float  # growth rate ΔDW/Δt (g/L/h)

def make_intervals(dw_df: pd.DataFrame) -> list[IntervalRow]:
    """Consecutive DW intervals and growth rate per hour."""
    rows = []
    for i in range(1, len(dw_df)):
        t0, t1 = dw_df.loc[i-1, TIME_COL], dw_df.loc[i, TIME_COL]
        dw0, dw1 = dw_df.loc[i-1, "Dryweight"], dw_df.loc[i, "Dryweight"]
        dt_h = (t1 - t0).total_seconds()/3600.0
        if dt_h <= 0:
            continue
        d_dw = dw1 - dw0
        g = d_dw / dt_h
        rows.append(IntervalRow(t0=t0, t1=t1, dt_h=dt_h, d_dw=d_dw, g=g))
    return rows

def aggregate_sensors_over_interval(sensor: pd.DataFrame, iv: IntervalRow) -> dict:
    """Aggregate each driver over (t0, t1] — past-only to avoid leakage."""
    mask = (sensor[TIME_COL] > iv.t0) & (sensor[TIME_COL] <= iv.t1)
    chunk = sensor.loc[mask, [TIME_COL] + [c for c in SENSOR_COLS if c in sensor.columns]].copy()
    # if no rows fall in the interval (rare), use nearest-prior sample
    if chunk.empty:
        prev_idx = sensor[sensor[TIME_COL] <= iv.t1].index.max()
        if pd.isna(prev_idx):
            return None
        chunk = sensor.loc[[prev_idx], [TIME_COL] + [c for c in SENSOR_COLS if c in sensor.columns]]

    feat = {}
    for c in [col for col in SENSOR_COLS if col in chunk.columns]:
        vec = pd.to_numeric(chunk[c], errors="coerce").dropna()
        if vec.empty:
            continue
        for name, fn in AGG_FUNCS.items():
            feat[f"{c}_{name}"] = fn(vec.values)
    # add simple “integral-like” features for flows over time
    if "FLOW.OF.CO2" in chunk.columns:
        flow_mean = pd.to_numeric(chunk["FLOW.OF.CO2"], errors="coerce").dropna().mean()
        feat["CO2_feed_approx"] = flow_mean * iv.dt_h  # (m/s * h) ~ proxy
    if "FLOW.OF.ALGAE" in chunk.columns:
        flow_a_mean = pd.to_numeric(chunk["FLOW.OF.ALGAE"], errors="coerce").dropna().mean()
        feat["ALGAE_flow_approx"] = flow_a_mean * iv.dt_h
    # include dt (some processes scale with interval length)
    feat["dt_h"] = iv.dt_h
    return feat

def build_training_table(sensor: pd.DataFrame, dw_df: pd.DataFrame) -> pd.DataFrame:
    intervals = make_intervals(dw_df)
    rows = []
    for iv in intervals:
        feats = aggregate_sensors_over_interval(sensor, iv)
        if feats is None:
            continue
        feats.update({
            "t0": iv.t0,
            "t1": iv.t1,
            "g_rate": iv.g,      # target
            "d_dw": iv.d_dw,     # for reference
            "dw0": dw_df.loc[dw_df[TIME_COL]==iv.t0, "Dryweight"].iloc[0],
            "dw1": dw_df.loc[dw_df[TIME_COL]==iv.t1, "Dryweight"].iloc[0],
        })
        rows.append(feats)
    df = pd.DataFrame(rows).dropna(axis=1, how="all")
    # keep only numeric features + target
    num = df.select_dtypes(include=[np.number]).copy()
    # drop degenerate columns
    nunique = num.nunique()
    num = num[nunique[nunique > 1].index]
    # >>> prevent leakage
    for leak in ["dw1", "d_dw"]:
        if leak in num.columns:
            num = num.drop(columns=[leak])
    return num

# ---------- MODEL ----------
def time_split(df_num: pd.DataFrame, target="g_rate", train_frac=0.8):
    df_sorted = df_num.sort_index()
    n = len(df_sorted)
    cut = int(n * train_frac)
    train = df_sorted.iloc[:cut]
    test  = df_sorted.iloc[cut:]
    Xtr = train.drop(columns=[target]).values
    ytr = train[target].values
    Xte = test.drop(columns=[target]).values
    yte = test[target].values
    cols = [c for c in train.columns if c != target]
    return (Xtr, ytr, Xte, yte, cols, train, test)


def build_gam(n_features: int, n_splines: int = 6):
    if n_features < 1:
        return LinearGAM()
    terms = s(0, n_splines=n_splines)
    for i in range(1, n_features):
        terms += s(i, n_splines=n_splines)
    return LinearGAM(terms)

def fit_gam(gam: LinearGAM, Xtr, ytr):
    lam_grid = np.logspace(0, 4, 8)  # heavier smoothing for tiny N
    return gam.gridsearch(Xtr, ytr, lam=lam_grid, progress=False)

def evaluate_gam(gam: LinearGAM, Xtr, ytr, Xte, yte):
    def mse(y, yhat): return float(np.mean((y - yhat)**2))
    yhat_tr = gam.predict(Xtr)
    yhat_te = gam.predict(Xte)
    return {
        "mse_train": mse(ytr, yhat_tr),
        "mse_test":  mse(yte, yhat_te),
        "r2_train":  float(1 - np.var(ytr - yhat_tr)/np.var(ytr)) if np.var(ytr)>0 else np.nan,
        "r2_test":   float(1 - np.var(yte - yhat_te)/np.var(yte)) if np.var(yte)>0 else np.nan,
    }

# ---------- SIMPLE FORWARD SIMULATOR ----------
def simulate(gam: LinearGAM, x_template: pd.Series, dw0: float, hours: float, step_h: float = 1.0):
    """
    Simulate DW trajectory with constant (or externally-updated) drivers.
    x_template: a pandas Series with feature values (same columns as training X),
                e.g., set PAR/T/pH/pCO2/flows to chosen setpoints; dt_h will be 'step_h'.
    """
    t = 0.0
    dw = dw0
    traj = [(t, dw)]
    cols = x_template.index.tolist()
    while t < hours - 1e-9:
        x = x_template.copy()
        if "dt_h" in cols: x["dt_h"] = step_h
        g_hat = float(gam.predict(x.values.reshape(1, -1))[0])
        dw = dw + g_hat * step_h
        t += step_h
        traj.append((t, dw))
    out = pd.DataFrame(traj, columns=["t_h","DW"])
    return out

# ---------- MAIN ----------
if __name__ == "__main__":
    sensors = read_sensor_zhaw(ZHAW_DATA_CSV)
    meta    = read_meta_zhaw(ZHAW_META_CSV)

    # keep only sensor columns we use
    have = [c for c in SENSOR_COLS if c in sensors.columns]
    sensors = sensors[[TIME_COL] + have].copy()

    data_num = build_training_table(sensors, meta)
    if "g_rate" not in data_num.columns or len(data_num) < 5:
        raise RuntimeError("Too few growth-interval rows; check DW points and file paths.")

    Xtr, ytr, Xte, yte, cols, train_df, test_df = time_split(data_num, target="g_rate", train_frac=0.8)
    gam = build_gam(n_features=len(cols))
    gam = fit_gam(gam, Xtr, ytr)
    metrics = evaluate_gam(gam, Xtr, ytr, Xte, yte)

    print(f"Training rows: {len(Xtr)}, Test rows: {len(Xte)}, Features: {len(cols)}")
    print("GAM metrics:", metrics)

    # Top effects (std dev of partial dependence over the grid)
    effects = {}
    for i, name in enumerate(cols):
        XX = gam.generate_X_grid(term=i)           # shape: (n_grid, n_features)
        pdp = gam.partial_dependence(term=i, X=XX) # vector length n_grid
        effects[name] = float(np.std(pdp))
    top = sorted(effects.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top effect drivers:", top)

    # Example simulation at median operating conditions from training set:
    x0 = train_df.drop(columns=["g_rate"]).median()
    sim = simulate(gam, x0, dw0=float(train_df["dw0"].median()), hours=72, step_h=1.0)
    print(sim.head())
