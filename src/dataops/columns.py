"""
列名映射、去除重复列、数值化
"""

from __future__ import annotations
import pandas as pd
import re

CN2STD = {
    # 代码/名称 常见写法都映射
    "代码": "symbol", "股票代码": "symbol", "证券代码": "symbol",
    "名称": "name", "证券简称": "name",
    # 常见数值列
    "收盘": "close", "收盘价": "close",
    "涨幅%": "pct_chg", "涨跌幅%": "pct_chg", "涨跌幅": "pct_chg",
    "成交量": "volume", "成交额": "amount",
    "涨跌额": "chg", "流通股": "float_shares",
    "委买均价": "bid_avg_px", "委卖均价": "ask_avg_px",
    # 其他保持不变，可继续补充
}


def _strip_unnamed(cols):
    """去除Unnamed列"""
    return [("" if str(c).startswith("Unnamed:") else str(c).strip()) for c in cols]


def lift_header_row_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    """如果第一行是说明，第二行才是表头，则第二行提升为表头。"""
    df = df.copy()
    cols = _strip_unnamed(df.columns)
    looks_like_title = not any(x in {"代码", "股票代码", "证券代码"} for x in cols[:3])

    if df.shape[0] >= 1:
        row0 = [str(x).strip() for x in df.iloc[0].tolist()]
        row0_has_header = any(x in {"代码", "股票代码", "证券代码"} for x in row0)

        if looks_like_title and row0_has_header:
            df.columns = row0
            df = df.iloc[1:].reset_index(drop=True)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名"""
    cols = []
    for col in df.columns:
        key = re.sub(r"\s+", "", str(col))  # 去除空白
        key = key.replace("％", "%")
        cols.append(CN2STD.get(key, key))  # 映射
    out = df.copy()
    out.columns = cols
    return out


def drop_head_dup(df: pd.DataFrame) -> pd.DataFrame:
    """去除重复列，保留最后一列"""
    if df.shape[0] >= 2:
        head = df.iloc[0].astype(str).tolist()
        if ("代码" in head) and ("名称" in head):
            return df.iloc[1:].reset_index(drop=True)
    return df


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False),
        errors="coerce"
    )


def standardize_sheet(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    - 处理两行表头：如有需要，把第二行提升为真正表头
    - 映射中文列名到标准英文
    - 数值化（除 symbol/name 以外）
    - 规范化 6 位代码，去掉全非数字
    - 丢弃全空列/全空行
    """
    df = df_raw.dropna(how="all").copy()

    # ① 提升第二行为表头（适配“历史行情…”+“代码/名称/…”结构）
    df = lift_header_row_if_needed(df)

    # ② 去掉完全空的列（大量 Unnamed）
    df = df.dropna(axis=1, how="all")
    df.columns = [str(c).strip() for c in df.columns]

    # ③ 列名标准化
    df = standardize_columns(df)

    # ④ 数值化（symbol/name 除外）
    for c in df.columns:
        if c not in {"symbol","name"}:
            df[c] = to_numeric_safe(df[c])

    # ⑤ 代码规范：仅数字 + 6 位左侧补零
    if "symbol" in df.columns:
        df["symbol"] = (
            df["symbol"].astype(str)
            .str.replace(r"\D","", regex=True)
            .str.zfill(6)
        )

    # ⑥ 丢弃完全空行
    df = df.dropna(how="all")

    return df
