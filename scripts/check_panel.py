import re
import pandas as pd


def check_invalid_symbols(df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    if symbol_col not in df.columns:
        raise KeyError(f"DataFrame 中不存在列 {symbol_col}")

    pattern = re.compile(r"^\d{6}(\.[A-Z]{2})?$")  # 600000 或 600000.SH
    mask = ~df[symbol_col].astype(str).str.match(pattern, na=False)

    return df.loc[mask].copy()


panel = pd.read_parquet("../data/processed/panel.parquet")

invalid_rows = check_invalid_symbols(panel)

if len(invalid_rows):
    print("发现异常symbol行：")
    print(invalid_rows.head(10))
else:
    print("所有symbol都符合规则")
