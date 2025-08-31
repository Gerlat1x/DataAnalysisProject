from __future__ import annotations

import numpy as np
import pandas as pd


class LinearRegressionModel:
    """
    一个简单的线性回归模型
    用 (open, high, low, volume_std, amount) 预测 close
    用于验证数据的输入输出格式是否正确
    """
    FEATURES = ["open", "high", "low", "volume_std", "amount"]

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, df: pd.DataFrame) -> "LinearRegressionModel":
        required = set(self.FEATURES + ["close"])
        if not required.issubset(df.columns):
            missing = required.difference(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        X = df[self.FEATURES].to_numpy(dtype=float)
        y = df["close"].to_numpy(dtype=float)

        X_design = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        required = set(self.FEATURES)
        if not required.issubset(df.columns):
            missing = required.difference(df.columns)
            raise ValueError(f"Missing columns: {missing}")

        X = df[self.FEATURES].to_numpy(dtype=float)
        y_pred = X @ self.coef_ + self.intercept_
        return pd.Series(y_pred, index=df.index, name="close")
