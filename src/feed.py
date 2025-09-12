""""
读取一个目录下的大量“日度截面excel”，标准化为一个回测面板
逐行输出（datatime，symbol，。。。）的DataFrame最为一个“barset”
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable

import pandas as pd

from src.dataops.columns import standardize_sheet
from src.dataops.ohlc import ensure_required_ohlc
from src.dataops.panel import attach_trade_date, validate_panel_schema
from src.infra.excel import read_excel_any
from src.infra.files import list_excels_sorted

BASE_COLS_ORDER = [
    "datetime", "symbol", "open", "high", "low", "close",
    "volume", "amount", "pre_close", "name",
]

DayFilter = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]
PanelFilter = Callable[[pd.DataFrame], pd.DataFrame]


def _pick_first_nonempty_sheet(df_or_dict) -> pd.DataFrame:
    """read_excel_any 可能返回 DataFrame 或 {sheet_name: DataFrame}；这里选择第一个非空表。"""
    if isinstance(df_or_dict, pd.DataFrame):
        return df_or_dict
    if isinstance(df_or_dict, dict):
        for v in df_or_dict.values():
            if isinstance(v, pd.DataFrame) and len(v.dropna(how="all")):
                return v
    # 都不合格时给出空表
    return pd.DataFrame()


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将 BASE_COLS_ORDER 放在前面，其余列保持相对顺序在后。"""
    base = [c for c in BASE_COLS_ORDER if c in df.columns]
    tail = [c for c in df.columns if c not in base]
    return df[base + tail]


@dataclass
class ExcelDailyFeed:
    """
        逐日读取与标准化 Excel 截面数据，并（可选）构建 panel。
        - data_dir: 数据目录
        - pattern: 文件匹配模式（默认 *.xls* 兼容 xls/xlsx）
        - sheet_name: 指定 sheet；None 时自动选择第一个非空 sheet
        - sort_by_symbol: 每日 barset 是否按 symbol 排序
        - day_filters: 对“每日 barset”的过滤器列表（在 attach_trade_date 之后执行）
                       形如 fn(df: DataFrame, trade_date: Timestamp) -> DataFrame
        - panel_filters: 拼接完成后对“整张 panel”的过滤器列表（如 IPO ≥ N 天）
        - logger: 可注入日志器；默认使用模块级 logger
        """
    data_dir: str
    pattern: str = "*.xls*"
    sheet_name: Optional[str] = None
    sort_by_symbol: bool = True
    day_filters: Optional[List[DayFilter]] = None
    panel_filters: Optional[List[PanelFilter]] = None
    logger: Optional[logging.Logger] = None

    def __post_init__(self) -> None:
        self.logger = self.logger or logging.getLogger(__name__)
        self.pairs: List[Tuple[pd.Timestamp, str]] = list_excels_sorted(self.data_dir, self.pattern)
        self._i = 0

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self) -> Tuple[pd.Timestamp, pd.DataFrame]:
        if self._i >= len(self.pairs):
            raise StopIteration
        trade_date, file_path = self.pairs[self._i]
        self._i += 1
        return trade_date, self._load_one(file_path, trade_date)

    def _load_one(self, file_path: str, trade_date: pd.Timestamp) -> pd.DataFrame:
        df_raw = read_excel_any(file_path, sheet_name=self.sheet_name)
        df_raw = _pick_first_nonempty_sheet(df_raw)

        if df_raw.empty:
            self.logger.warning(f"[{trade_date.date()}] 读取文件 {file_path} 为空，跳过")
            return pd.DataFrame(columns=BASE_COLS_ORDER)

        df = standardize_sheet(df_raw)

        if "symbol" not in df.columns:
            # 打印原始列名帮助定位（可临时保留，排查完再移除）
            raise ValueError(
                f"[{file_path}] 解析失败：未找到 'symbol' 列。"
                f" 原始列名示例={list(df_raw.columns)[:8]}, 标准化后列名示例={list(df.columns)[:8]}"
            )

        df = df.dropna(subset=["symbol"])
        df = ensure_required_ohlc(df)
        df = attach_trade_date(df, trade_date)

        df = _reorder_columns(df)

        if self.day_filters:
            for f in self.day_filters:
                before = len(df)
                df = f(df, trade_date)
                if not isinstance(df, pd.DataFrame):
                    raise TypeError(f"day_filter 返回类型必须是 DataFrame，实际是 {type(df)}")
                self.logger.debug("[DayFilter] %s -> %d 行 (from %d) @ %s",
                                  f.__name__ if hasattr(f, "__name__") else str(f), len(df), before, trade_date)

        if self.sort_by_symbol and "symbol" in df.columns:
            df = df.sort_values("symbol").reset_index(drop=True)

        return df

    def build_panel(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for trade_date, file_path in self.pairs:
            try:
                frames.append(self._load_one(file_path, trade_date))
            except Exception:
                self.logger.exception("[ExcelDailyFeed] 解析失败：%s @ %s", file_path, trade_date)
                continue

        if not frames:
            raise RuntimeError("未生成任何日度 barset，检查数据目录或解析逻辑。")

        panel = pd.concat(frames, ignore_index=True)

        if self.panel_filters:
            for f in self.panel_filters:
                before = len(panel)
                panel = f(panel)
                if not isinstance(panel, pd.DataFrame):
                    raise TypeError(f"panel_filter 返回类型必须是 DataFrame，实际是 {type(panel)}")
                self.logger.debug("[PanelFilter] %s -> %d 行 (from %d)",
                                  f.__name__ if hasattr(f, "__name__") else str(f), len(panel), before)

        # 排序与校验
        sort_cols = [c for c in ("datetime", "symbol") if c in panel.columns]
        if sort_cols:
            panel = panel.sort_values(sort_cols).reset_index(drop=True)

        validate_panel_schema(panel)
        return panel
