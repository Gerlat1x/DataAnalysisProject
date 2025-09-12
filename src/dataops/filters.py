import re
import pandas as pd

_ST_PREFIX_RE = re.compile(r"(?i)^\s*(?:\*+|S\*)?ST(?:[\s\-（(]|$)")

# 风险/退市关键词（包含时直接剔除）
RISK_KEYWORDS = (
    "退市", "退市整理", "风险警示", "暂停上市", "终止上市"
)


def is_st_like_name(name: str) -> bool:
    if not isinstance(name, str):
        return False
    s = name.strip().upper().replace("＊", "*").replace("ＳＴ", "ST")
    if "ST" in s:  # 只要包含 ST / *ST 就标记
        return True
    for kw in RISK_KEYWORDS:
        if kw in s:
            return True
    return False


def mark_st_flags(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """返回带 is_st 列的副本；若没有 name 列，直接返回原表。"""
    if name_col not in df.columns:
        return df
    df_marked = df.copy()
    df_marked["is_st"] = df_marked[name_col].apply(is_st_like_name)
    return df_marked


def filter_new_stocks(df: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
    ipo_dates = df.groupby("symbol")["date"].min()
    df = df.merge(ipo_dates.rename("ipo_date"), on="symbol")
    df["days_since_ipo"] = (df["date"] - df["ipo_date"]).dt.days
    filtered = df[df["days_since_ipo"] >= min_days].copy()
    return filtered


def apply_all_filters(
    df: pd.DataFrame,
    min_days: int = 60,
    drop_st: bool = True,
    name_col: str = "name",
) -> pd.DataFrame:
    """
    统一入口：应用所有过滤规则（ST、新股等）
    """
    df_filtered = df.copy()

    # ST 标记
    df_filtered = mark_st_flags(df_filtered, name_col=name_col)

    # 剔除 ST
    if drop_st and "is_st" in df_filtered.columns:
        df_filtered = df_filtered[~df_filtered["is_st"]].copy()

    # 新股过滤
    df_filtered = filter_new_stocks(df_filtered, min_days=min_days)

    return df_filtered
