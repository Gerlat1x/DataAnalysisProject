""""
纯算法解析日期字符串为时间戳（无I/O）
"""
from __future__ import annotations
import re
from typing import Optional

_PATTERN = [
    r"(20\d{2})[-_/\.]?(\d{1,2})[-_/\.]?(\d{1,2})",
    r"(20\d{2})年(\d{1,2})月(\d{1,2})日",
]


def infer_date_from_text(text: str) -> Optional[str]:
    """从文本中推断出日期字符串，格式为YYYY-MM-DD"""
    for pattern in _PATTERN:
        match = re.search(pattern, text)
        if match:
            year, month, day = map(int, match.groups())
            return f"{year:04d}-{month:02d}-{day:02d}"
    return None
