""""
与Excel交互，禁止业务逻辑与数据清洗规则
"""
from __future__ import annotations
import pandas as pd


def read_excel_any(path: str, sheet_name=None) -> pd.DataFrame | dict[str, pd.DataFrame]:
    # 先用openpyxl，失败用xlrd
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        try:
            return pd.read_excel(path, sheet_name=sheet_name, engine="xlrd")
        except Exception:
            return pd.read_excel(path, sheet_name=sheet_name)
