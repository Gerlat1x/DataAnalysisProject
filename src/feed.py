""""
读取一个目录下的大量“日度截面excel”，标准化为一个回测面板
逐行输出（datatime，symbol，。。。）的DataFrame最为一个“barset”
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import pandas as pd
from src.infra.excel import read_excel_any
from src.infra.files import list_excels_sorted
from src.dataops.ohlc import ensure_required_ohlc
from src.dataops.panel import attach_trade_date, validate_panel_schema
from src.dataops.columns import standardize_sheet


class ExcelDailyFeed:
    def __init__(self, data_dir: str, pattern: str = "*.xls*", sheet_name: Optional[str] = None, sort_by_symbol=True):
        self.pairs = list_excels_sorted(data_dir, pattern)  # [(date, path), ...]
        self.sheet_name = sheet_name
        self.sort_by_symbol = sort_by_symbol
        self._i = 0

    def __iter__(self):
        self._i = 0;
        return self

    def __next__(self) -> Tuple[pd.Timestamp, pd.DataFrame]:
        if self._i >= len(self.pairs): raise StopIteration
        d, fp = self.pairs[self._i];
        self._i += 1
        return d, self._load_one(fp, d)

    def _load_one(self, file_path: str, trade_date: pd.Timestamp) -> pd.DataFrame:
        df_raw = read_excel_any(file_path, sheet_name=self.sheet_name)
        if isinstance(df_raw, dict):  # 多sheet时取首个非空
            for v in df_raw.values():
                if isinstance(v, pd.DataFrame) and len(v.dropna(how="all")):
                    df_raw = v;
                    break
        df = standardize_sheet(df_raw).dropna(subset=["symbol"])
        df = ensure_required_ohlc(df)
        df = attach_trade_date(df, trade_date)
        # 最小列集合 + 其余特征
        base = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "pre_close", "name"]
        base = [c for c in base if c in df.columns]
        df = df[base + [c for c in df.columns if c not in base]]
        if "symbol" not in df.columns:
            # 打印原始列名帮助定位（可临时保留，排查完再移除）
            raise ValueError(
                f"[{file_path}] 解析失败：未找到 'symbol' 列。"
                f" 原始列名示例={list(df_raw.columns)[:8]}, 提升后列名示例={list(df.columns)[:8]}"
            )

        df = df.dropna(subset=["symbol"])
        if self.sort_by_symbol and "symbol" in df.columns:
            df = df.sort_values("symbol").reset_index(drop=True)
        return df

    def build_panel(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for d, fp in self.pairs:
            frames.append(self._load_one(fp, d))
        panel = pd.concat(frames, ignore_index=True).sort_values(["datetime", "symbol"])
        validate_panel_schema(panel)
        return panel.reset_index(drop=True)
