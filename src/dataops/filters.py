from __future__ import annotations
import re
import pandas as pd
from typing import Optional, Callable, List, Dict

_ST_PREFIX_RE = re.compile(r"(?i)^\s*(?:\*+|S\*)?ST(?:[\s\-（(]|$)")

# 风险/退市关键词（包含时直接剔除）
RISK_KEYWORDS = (
    "退市", "退市整理", "风险警示", "暂停上市", "终止上市"
)


def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # 全角/半角与常见替换
    s = s.replace("＊", "*").replace("ＳＴ", "ST")
    return s.strip()


def is_st_like_name(name: str, risk_keywords: tuple = RISK_KEYWORDS) -> bool:
    s = normalize_name(name).upper()
    if "ST" in s:  # 只要包含 ST / *ST 就标记
        return True
    if _ST_PREFIX_RE.match(s):  # 以 ST/*ST/S* 开头的
        return True
    for kw in risk_keywords:
        if kw in s:
            return True
    return False
    # if not isinstance(name, str):
    #     return False
    # s = name.strip().upper().replace("＊", "*").replace("ＳＴ", "ST")
    # if "ST" in s:  # 只要包含 ST / *ST 就标记
    #     return True
    # for kw in RISK_KEYWORDS:
    #     if kw in s:
    #         return True
    # return False


def mark_st_flags(df: pd.DataFrame,
                  name_col: str = "name",
                  out_col: str = "is_st",
                  risk_keywords: tuple = RISK_KEYWORDS,
                  ) -> pd.DataFrame:
    """返回带 is_st 列的副本；若没有 name 列，直接返回原表。"""
    if name_col not in df.columns:
        return df.copy()
    df_marked = df.copy()
    df_marked[out_col] = df_marked[name_col].apply(
        lambda x: is_st_like_name(x, risk_keywords=risk_keywords)
    )
    return df_marked


def filter_by_ipo_days(
        df: pd.DataFrame,
        min_days: int = 60,
        date_col: str = "datetime",
        symbol_col: str = "symbol",
        keep_aux_cols: bool = False,
) -> pd.DataFrame:
    """过滤上市未满 min_days 的记录"""
    if date_col not in df.columns or symbol_col not in df.columns:
        return df.copy()
    res = df.copy()
    ipo_dates = res.groupby(symbol_col)[date_col].min()
    res = res.merge(ipo_dates.rename("ipo_date"), on=symbol_col)
    res["days_since_ipo"] = (res[date_col] - res["ipo_date"]).dt.days
    res = res[res["days_since_ipo"] >= min_days].copy()
    if not keep_aux_cols:
        res.drop(columns=["ipo_date", "days_since_ipo"], inplace=True, errors="ignore")
    return res


# def filter_new_stocks(df: pd.DataFrame, min_days: int = 60) -> pd.DataFrame:
#     ipo_dates = df.groupby("symbol")["date"].min()
#     df = df.merge(ipo_dates.rename("ipo_date"), on="symbol")
#     df["days_since_ipo"] = (df["date"] - df["ipo_date"]).dt.days
#     filtered = df[df["days_since_ipo"] >= min_days].copy()
#     return filtered

def apply_blacklist(
        df: pd.DataFrame,
        symbols: List[str],
        symbol_col: str = "symbol",
) -> pd.DataFrame:
    """剔除黑名单中的股票"""
    if not symbols or symbol_col not in df.columns:
        return df.copy()
    return df[~df[symbol_col].isin(symbols)].copy()


def apply_whitelist(
        df: pd.DataFrame,
        symbols: List[str],
        symbol_col: str = "symbol",
) -> pd.DataFrame:
    """仅保留白名单中的股票"""
    if not symbols or symbol_col not in df.columns:
        return df.copy()
    return df[df[symbol_col].isin(symbols)].copy()


"""
↓↓↓↓管道类↓↓↓↓
"""

Rule = Callable[[pd.DataFrame], pd.DataFrame]


class FilterPipeline:
    def __init__(self):
        self._rules: List[Rule] = []
        self._mata: Dict[str, object] = {}

    def add_rule(self, rule: Rule) -> "FilterPipeline":
        self._rules.append(rule)
        return self

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for rule in self._rules:
            out = rule(out)
        return out

    """
    ↓↓↓↓内置规则↓↓↓↓
    """

    def mark_st(
            self,
            name_col: str = "name",
            out_col: str = "is_st",
            risk_keywords: tuple = RISK_KEYWORDS,
    ) -> "FilterPipeline":
        def _rule(df: pd.DataFrame) -> pd.DataFrame:
            return mark_st_flags(df, name_col=name_col, out_col=out_col, risk_keywords=risk_keywords)

        return self.add_rule(_rule)

    def drop_st(self, flag_col: str = "is_st") -> "FilterPipeline":
        def _rule(df: pd.DataFrame) -> pd.DataFrame:
            if flag_col not in df.columns:
                # 若未标记，兼容性兜底：尝试即时标记
                df_tmp = mark_st_flags(df)
            else:
                df_tmp = df
            return df_tmp[~df_tmp.get(flag_col, False)].copy()

        return self.add_rule(_rule)

    def require_ipo_days(
            self,
            min_days: int = 60,
            date_col: str = "date",
            symbol_col: str = "symbol",
            keep_aux_cols: bool = False,
    ) -> "FilterPipeline":
        def _rule(df: pd.DataFrame) -> pd.DataFrame:
            return filter_by_ipo_days(
                df,
                min_days=min_days,
                date_col=date_col,
                symbol_col=symbol_col,
                keep_aux_cols=keep_aux_cols,
            )

        return self.add_rule(_rule)

    def blacklist(
            self,
            symbols: List[str],
            symbol_col: str = "symbol",
    ) -> "FilterPipeline":
        def _rule(df: pd.DataFrame) -> pd.DataFrame:
            return apply_blacklist(df, symbols, symbol_col=symbol_col)

        return self.add_rule(_rule)

    def whitelist(
            self,
            symbols: List[str],
            symbol_col: str = "symbol",
    ) -> "FilterPipeline":
        def _rule(df: pd.DataFrame) -> pd.DataFrame:
            return apply_whitelist(df, symbols, symbol_col=symbol_col)

        return self.add_rule(_rule)
