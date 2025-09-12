"""
处理数据，作为二分类问题的输入
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, List


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    """安全地将字符串转换为日期时间，无法解析的返回 NaT"""
    return pd.to_datetime(s, errors="coerce")


def _compute_rsi(ret: pd.Series, period: int = 14) -> pd.Series:
    """
    计算相对强弱指数 (RSI)
    Args:
        ret (pd.Series): 过去的收益率，防止泄露未来信息给模型
    """
    gain = ret.clip(lower=0)
    loss = (-ret).clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def make_binary_dataset(
        panel: pd.DataFrame,
        window: int = 50,
        date_col: str = "datetime",
        symbol_col: str = "symbol",
        price_cols: Optional[List[str]] = None,
        volume_col: str = "volume",
) -> pd.DataFrame:
    """
    用 t-50..t-1 的窗口构造特征，预测 t 日“收盘较前一日是否上涨”（label ∈ {0,1}）。

    期望输入（长表）至少包含：
      - datetime / symbol / close / (open, high, low, volume 可选但强烈建议有)

    返回：带特征与 label 的长表，已按 (datetime, symbol) 排序，去掉了前 window 天样本。
    """
    if price_cols is None:
        price_cols = ["open", "high", "low", "close"]

    required = {date_col, symbol_col, "close"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")

    df = panel.copy()
    df[date_col] = _safe_to_datetime(df[date_col])
    df = df.sort_values([symbol_col, date_col])

    def _per_symbol(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values(date_col).copy()
        g["label"] = (g["close"] > g["close"].shift(1)).astype("Int64")

        past_close = g["close"].shift(1)
        past_open = g["open"].shift(1) if "open" in g.columns else None
        past_high = g["high"].shift(1) if "high" in g.columns else None
        past_low = g["low"].shift(1) if "low" in g.columns else None
        past_vol = g[volume_col].shift(1) if volume_col in g.columns else None
        past_ma_close_w = past_close.rolling(window, min_periods=window).mean()
        past_ret_1d = past_close / past_close.shift(1) - 1

        roll_ret = past_ret_1d.rolling(window, min_periods=window)
        roll_close = past_close.rolling(window, min_periods=window)
        roll_vol = past_vol.rolling(window, min_periods=window) if past_vol is not None else None

        g["f_ret_mean_w"] = roll_ret.mean()
        g["f_ret_std_w"] = roll_ret.std()

        g["f_ma_close_w"] = roll_close.mean()
        # g["f_close_over_ma"] = g["close"] / g["f_ma_close_w"]
        g["f_close_over_ma_prev"] = past_close / past_ma_close_w

        for k in (5, 10, 20):
            g[f"f_mom_{k}"] = past_close / past_close.shift(k) - 1

        if past_vol is not None:
            g["f_vol_ma_w"] = roll_vol.mean()
            g["f_vol_ratio"] = past_vol / g["f_vol_ma_w"]  # 量比（昨天 vs 过去均量）
        else:
            g["f_vol_ma_w"] = np.nan
            g["f_vol_ratio"] = np.nan

        if all(c in g.columns for c in ["open", "high", "low"]):
            # 昨日振幅（相对昨日收盘）
            if past_high is not None and past_low is not None:
                g["f_amp_prev"] = (past_high - past_low) / past_close
            else:
                g["f_amp_prev"] = np.nan

            # 今日跳空（开盘 vs 昨收）——注意：这会用到“今日 open”和“昨日 close”
            # 对于“预测 t 日”的模型，今日开盘是已知（盘前/开盘后），可按你的使用场景决定是否纳入
            g["f_gap_open"] = (g["open"] - g["close"].shift(1)) / g["close"].shift(1)
        else:
            g["f_amp_prev"] = np.nan
            g["f_gap_open"] = np.nan

        g["f_rsi_14"] = _compute_rsi(past_ret_1d, period=14)

        # 布林带带宽（基于过去）
        g["f_bb_mean"] = roll_close.mean()
        g["f_bb_std"] = roll_close.std()
        g["f_bb_width"] = (2 * g["f_bb_std"]) / g["f_bb_mean"]  # 近似宽度/均值

        # 丢弃前 window 行（特征不完整）
        g = g.dropna(subset=["f_ret_mean_w", "f_ret_std_w", "f_ma_close_w"]).copy()

        # 只保留我们需要的列（保留原始关键列，便于回查）
        keep = [c for c in [
            date_col, symbol_col, "open", "high", "low", "close", volume_col
        ] if c in g.columns]
        feats = [c for c in g.columns if c.startswith("f_")]
        return g[[date_col, symbol_col] + feats + ["label"]]

    parts = []
    for sym, g in df.groupby(symbol_col, group_keys=False):
        part = _per_symbol(g)
        # 若 _per_symbol 返回时没带 symbol 列，这里补上
        if symbol_col not in part.columns:
            part[symbol_col] = sym
        parts.append(part)

    out = (pd.concat(parts, ignore_index=True)
           .sort_values([date_col, symbol_col])
           .reset_index(drop=True))
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out
