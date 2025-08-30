"""
补齐 OHLC/pre_close 数据
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ensure_required_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pct_chg" in df.columns and "close" in df.columns and "pre_close" not in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["pre_close"] = df["close"] / (1.0 + df["pct_chg"] / 100.0)
    if "open" not in df.columns:
        df["open"] = df.get("pre_close", df.get("close"))
    if "high" not in df.columns:
        df["high"] = np.maximum(df["open"].values, df["close"].values)
    if "low" not in df.columns:
        df["low"] = np.minimum(df["open"].values, df["close"].values)
    for c in ["volume", "amount"]:
        if c not in df.columns:
            df[c] = 0.0
    return df
