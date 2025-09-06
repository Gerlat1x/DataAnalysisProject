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
    if "ST" in s:   # 只要包含 ST / *ST 就标记
        return True
    for kw in RISK_KEYWORDS:
        if kw in s:
            return True
    return False


def mark_st_flags(df: pd.DataFrame, name_col: str = "name") -> pd.DataFrame:
    """返回带 is_st 列的副本；若没有 name 列，直接返回原表。"""
    if name_col not in df.columns:
        return df
    out = df.copy()
    out["is_st"] = out[name_col].apply(is_st_like_name)
    return out
