"""
如果模型不能超过这个 baseline，那就没什么意义了
"""
import pandas as pd
import numpy as np


def baseline_naive(df: pd.DataFrame, group_col: str = "symbol") -> pd.DataFrame:
    """
    最简单的 baseline：预测今天 close = 昨天 close
    """
    df = df.sort_values([group_col, "datetime"]).copy()
    df["close_pred"] = df.groupby(group_col)["close"].shift(1)
    return df.dropna(subset=["close", "close_pred"])


def evaluate_baseline(df: pd.DataFrame) -> dict:
    """
    计算评估指标（和 LinearRegressionModel.evaluate 保持一致）
    """
    y_true = df["close"].to_numpy(dtype=float)
    y_pred = df["close_pred"].to_numpy(dtype=float)

    err = y_true - y_pred
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / np.clip(np.abs(y_true), 1e-12, None))) * 100.0
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n_eval": int(len(y_true)),
        "R2": float(r2),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE_%": float(mape),
    }

