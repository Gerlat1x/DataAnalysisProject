"""
文件系统操作（列目录、排序、缓存）
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from src.utils.dates import infer_date_from_text


def list_excels_sorted(directory: str, pattern: str = "*.xlsx"):
    """列出目录下所有Excel文件，并按文件名中的日期排序"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files: raise FileNotFoundError(f"No files found in {directory} with pattern {pattern}")
    pairs = []
    for file in files:
        date_str = infer_date_from_text(os.path.basename(file))
        if date_str is None:
            pairs.append((pd.NaT, file))
        else:
            pairs.append((pd.Timestamp(date_str).normalize(), file))

    pairs.sort(key=lambda x: (x[0], x[1]))
    return pairs
