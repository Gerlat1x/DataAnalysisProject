"""
附加日期、拼接面板、校验Schema
"""
from __future__ import annotations
import pandas as pd


def attach_trade_date(
        df: pd.DataFrame, trade_date: pd.Timestamp
) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "datetime", pd.Timestamp(trade_date).normalize())
    return df


def validate_panel_schema(df: pd.DataFrame, extra=None):
    need = {"datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "pre_close"}
    if extra: need |= set(extra)
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"DataFrame 缺少必要的列: {miss}")
