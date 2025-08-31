from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple


class LinearRegressionModel:
    """
    一个简单且可自检的线性回归基线
    用 (open, high, low, volume, amount) 预测 close
    """
    FEATURES = ["open", "high", "low", "volume", "amount"]

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._fitted_cols: list[str] | None = None
        self._n_train_: int | None = None

    def _require_cols(self, df: pd.DataFrame, cols: list[str]) -> None:
        missing = set(cols).difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

    def _prep_Xy(self, df: pd.DataFrame, need_y: bool) -> tuple[np.ndarray, np.ndarray | None, pd.Index]:
        cols = self.FEATURES + (["close"] if need_y else [])
        self._require_cols(df, cols)

        use = df[cols].copy()
        use = use.dropna(axis=0, how="any")
        idx = use.index

        X = use[self.FEATURES].to_numpy(dtype=float)
        y = use["close"].to_numpy(dtype=float) if need_y else None
        return X, y, idx

    def fit(self, df: pd.DataFrame) -> "LinearRegressionModel":
        X, y, _ = self._prep_Xy(df, need_y=True)

        # 设计矩阵 [1, X]
        X_design = np.hstack([np.ones((len(X), 1)), X])
        beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        self._fitted_cols = self.FEATURES.copy()
        self._n_train_ = len(X)
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")

        X, _, idx = self._prep_Xy(df, need_y=False)
        y_pred = X @ self.coef_ + self.intercept_
        # 用输入 df 的索引对齐；缺失被丢弃的行不返回预测
        return pd.Series(y_pred, index=idx, name="close_pred")

    def score(self, df: pd.DataFrame) -> float:
        """R²"""
        self._require_cols(df, self.FEATURES + ["close"])
        pred = self.predict(df)
        y_true = df.loc[pred.index, "close"].to_numpy(dtype=float)
        y_pred = pred.to_numpy(dtype=float)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    def residuals(self, df: pd.DataFrame) -> pd.Series:
        """返回残差：y_true - y_pred"""
        self._require_cols(df, self.FEATURES + ["close"])
        pred = self.predict(df)
        res = df.loc[pred.index, "close"].to_numpy(dtype=float) - pred.to_numpy(dtype=float)
        return pd.Series(res, index=pred.index, name="residual")

    def evaluate(self, df: pd.DataFrame) -> dict:
        """返回常用指标：R2 / MSE / RMSE / MAE / MAPE"""
        self._require_cols(df, self.FEATURES + ["close"])
        pred = self.predict(df)
        y_true = df.loc[pred.index, "close"].to_numpy(dtype=float)
        y_pred = pred.to_numpy(dtype=float)

        err = y_true - y_pred
        mse = float(np.mean(err ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(err)))
        # 防止除零
        mape = float(np.mean(np.abs(err / np.clip(np.abs(y_true), 1e-12, None)))) * 100.0
        r2 = self.score(df)

        return {
            "n_train": self._n_train_,
            "n_eval": int(len(y_true)),
            "R2": float(r2),
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE_%": mape,
            "intercept": self.intercept_,
            "coef": dict(zip(self.FEATURES, map(float, self.coef_))),
        }


class SafeLinearRegressionPipeline:
    """
    包装 LinearRegressionModel 的安全预测版
    - 自动生成滞后特征 (lag1)
    - 按时间切分 (train/test)
    """

    def __init__(self, base_model_cls, features: List[str], target: str = "close"):
        self.base_model_cls = base_model_cls
        self.features = features
        self.target = target
        self.model = None
        self.train_idx = None
        self.test_idx = None

    def add_lag_features(self, df: pd.DataFrame, group_col: str = "symbol", lag: int = 1) -> pd.DataFrame:
        df = df.sort_values([group_col, "datetime"]).copy()
        for f in self.features:
            df[f"{f}_lag{lag}"] = df.groupby(group_col)[f].shift(lag)
        return df

    def time_split(self, df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df.sort_values(["symbol", "datetime"])
        n = len(df)
        cut = int(n * train_frac)
        train, test = df.iloc[:cut], df.iloc[cut:]
        self.train_idx, self.test_idx = train.index, test.index
        return train, test

    def fit(self, df: pd.DataFrame, train_frac: float = 0.8) -> "SafeLinearRegressionPipeline":
        # 生成 lag 特征
        df = self.add_lag_features(df)
        lag_features = [f"{f}_lag1" for f in self.features]

        # 去掉 NaN（lag 产生的）
        df = df.dropna(subset=lag_features + [self.target])

        # 按时间切分
        train, test = self.time_split(df, train_frac)

        # 拟合模型
        self.model = self.base_model_cls()
        self.model.FEATURES = lag_features
        self.model.fit(train)

        return self

    def evaluate(self, df: pd.DataFrame) -> dict:
        # 用训练好的模型在测试集上评估
        df = self.add_lag_features(df)
        lag_features = [f"{f}_lag1" for f in self.features]
        df = df.dropna(subset=lag_features + [self.target])

        if self.test_idx is None:
            raise RuntimeError("You must fit() before evaluate()")

        test = df.loc[self.test_idx]
        return self.model.evaluate(test)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df = self.add_lag_features(df)
        lag_features = [f"{f}_lag1" for f in self.features]
        df = df.dropna(subset=lag_features)

        return self.model.predict(df)
