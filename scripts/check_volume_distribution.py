# check_volume_distribution.py (robust)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, anderson, skew, kurtosis

PATH = "../data/processed/panel.parquet"

# 1) Load
df = pd.read_parquet(PATH)
if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

# 2) Pick volume column
vol_col = "volume" if "volume" in df.columns else ("volume_std" if "volume_std" in df.columns else None)
if vol_col is None:
    raise ValueError("Neither 'volume' nor 'volume_std' found in dataframe.")


# 3) Choose a liquid symbol (recent 250d have >0 & non-null volume)
def pick_symbol(data: pd.DataFrame, vol_col: str, lookback: int = 250) -> str:
    # compute trailing window end by symbol
    g = data.groupby("symbol", group_keys=False)
    # last 250 rows per symbol
    tail_idx = g.tail(lookback).index
    tail = data.loc[tail_idx, ["symbol", vol_col]]
    ok = tail[vol_col].fillna(0) > 0
    # count positive-volume days
    counts = tail[ok].groupby("symbol")[vol_col].size().sort_values(ascending=False)
    if counts.empty:
        # fallback: overall positive count
        counts = data[data[vol_col].fillna(0) > 0].groupby("symbol")[vol_col].size().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No positive volume found for any symbol. Check your data.")
    return counts.index[0]


symbol = pick_symbol(df, vol_col, 250)
sample = df[df["symbol"] == symbol].copy()
sample = sample.sort_values("datetime").tail(250)

# 4) Build arrays (filter positive & non-null)
vol_raw = sample[vol_col].replace([np.inf, -np.inf], np.nan).dropna()
vol_raw = vol_raw[vol_raw > 0]
vol_log = np.log1p(vol_raw)

print(f"[Info] Using symbol: {symbol}")
print(f"[Info] Rows in last 250d for symbol: {len(sample)}")
print(f"[Info] Non-null positive {vol_col} count: {len(vol_raw)}")
print(f"[Info] Example stats - raw: mean={vol_raw.mean():.2f}, std={vol_raw.std():.2f}")
print(f"[Info] Example stats - log1p: mean={vol_log.mean():.4f}, std={vol_log.std():.4f}")
print("skew(raw), kurt(raw):", skew(vol_raw), kurtosis(vol_raw))
print("skew(log), kurt(log):", skew(vol_log), kurtosis(vol_log))

print("Shapiro raw:", shapiro(vol_raw))  # n=250 可用
print("Shapiro log:", shapiro(vol_log))

print("Anderson raw:", anderson(vol_raw, dist='norm'))
print("Anderson log:", anderson(vol_log, dist='norm'))


def volume_log_summary(df, vol_col="volume", lookback=250, top_n=20):
    out = []
    for sym, g in df.groupby("symbol"):
        s = g.sort_values("datetime").tail(lookback)[vol_col].dropna()
        s = s[s > 0]
        if len(s) < lookback // 2:  # 跳过过少数据
            continue
        ls = np.log1p(s)
        out.append({
            "symbol": sym,
            "n": len(s),
            "skew_raw": float(skew(s)),
            "skew_log": float(skew(ls)),
            "kurt_raw": float(kurtosis(s)),
            "kurt_log": float(kurtosis(ls)),
        })
    return pd.DataFrame(out).sort_values("skew_log").head(top_n)


summary = volume_log_summary(df, vol_col="volume")  # 或 "volume_std"
print(summary.head(10))
if len(vol_raw) == 0:
    raise ValueError(
        "Selected symbol has no positive volume in the last 250 days. Try increasing lookback or check data integrity.")

# 5) Histograms
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(vol_raw, bins=50, edgecolor="k")
plt.title(f"{symbol} Raw {vol_col} Distribution")
plt.xlabel(vol_col)
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(vol_log, bins=50, edgecolor="k")
plt.title(f"{symbol} log1p({vol_col}) Distribution")
plt.xlabel(f"log1p({vol_col})")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# 6) Q-Q plots
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
stats.probplot(vol_raw, dist="norm", plot=plt)
plt.title(f"{symbol} Raw {vol_col} Q-Q Plot")

plt.subplot(1, 2, 2)
stats.probplot(vol_log, dist="norm", plot=plt)
plt.title(f"{symbol} log1p({vol_col}) Q-Q Plot")

plt.tight_layout()
plt.show()
