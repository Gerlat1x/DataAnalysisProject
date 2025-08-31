import pandas as pd


def validate_constraints(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    对 panel 数据进行约束检查，返回违规情况统计
    """
    issues = {}

    # 1. 价格非负
    price_cols = ["open", "high", "low", "close", "pre_close"]
    for col in price_cols:
        issues[f"{col}_negative"] = int((df[col] < 0).sum())

    # 2. 成交量 / 成交额非负
    issues["volume_negative"] = int((df["volume"] < 0).sum())
    if "volume_std" in df.columns:
        issues["volume_std_negative"] = int((df["volume_std"] < 0).sum())
    issues["amount_negative"] = int((df["amount"] < 0).sum())

    # 3. OHLC 边界关系: low ≤ min(open, close) ≤ high
    bad_ohlc = df[
        (df["low"] > df[["open", "close"]].min(axis=1)) |
        (df["high"] < df[["open", "close"]].max(axis=1))
        ]
    issues["ohlc_violation"] = len(bad_ohlc)

    # 4. pre_close 对齐: pre_close ≈ 上一日 close
    if "symbol" in df.columns and "datetime" in df.columns:
        prev_close = df.groupby("symbol")["close"].shift(1)
        diff = (df["pre_close"] - prev_close).abs()
        issues["pre_close_mismatch"] = int((diff > 1e-6).sum())

    # 5. 极端成交量 / 成交额 (超过5σ)
    for col in ["volume", "amount"]:
        mean, std = df[col].mean(), df[col].std()
        if std > 0:
            outliers = ((df[col] - mean).abs() > 5 * std).sum()
            issues[f"{col}_5sigma_outlier"] = int(outliers)

    if verbose:
        print("==== 约束规则检查结果 ====")
        for k, v in issues.items():
            print(f"{k:25s}: {v}")

    return issues


if __name__ == "__main__":
    df = pd.read_parquet("../data/processed/panel.parquet")
    issues = validate_constraints(df)
    print(issues)
