from __future__ import annotations

"""
随机森林时间序列预测流水线（安全版）
- 自动生成滞后特征（避免未来信息泄漏）
- 可选的滚动统计特征与对数变换
- 按时间顺序划分训练集/测试集
- 提供多种评估指标（R2, MSE, RMSE, MAE, MAPE）
- 自带 Naive baseline（昨天收盘价预测今天收盘价）对比

假设输入数据 DataFrame 至少包含以下列：
['datetime','symbol','open','high','low','close','volume','amount']
如果你已经把成交量统一到“股”，请放在 'volume_std' 列里，否则将回退使用 'volume'。

使用示例：
---------------
from random_forest_pipeline import SafeRandomForestPipeline

features = ["open","high","low","volume_std","amount"]
p = SafeRandomForestPipeline(features=features,
                             target="close",
                             lags=[1],
                             add_rolling=True,
                             rolling_windows=(5, 20),
                             clip_outliers=True,
                             outlier_q=(0.01, 0.99),
                             log_volume=True,
                             log_amount=True)

p.fit(df, train_frac=0.8)  # 按时间顺序划分训练/测试
print(p.metrics_)          # 随机森林在测试集上的指标
print(p.baseline_metrics_) # Naive baseline 指标
pred = p.predict(df.tail(20))
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ------------------------- 工具函数 -------------------------

def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = set(cols).difference(df.columns)
    if missing:
        raise ValueError(f"缺失列: {missing}")


def _winsorize(df: pd.DataFrame, cols: Iterable[str], q_low: float, q_high: float) -> pd.DataFrame:
    """对指定列进行缩尾处理（按分位数裁剪极端值）"""
    df = df.copy()
    for c in cols:
        lo, hi = df[c].quantile([q_low, q_high])
        df[c] = df[c].clip(lo, hi)
    return df


def _log1p_cols(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """对数变换（log1p），防止极端数值主导模型"""
    df = df.copy()
    for c in cols:
        df[c] = np.log1p(np.clip(df[c].astype(float), a_min=0, a_max=None))
    return df


def _add_group_lags(df: pd.DataFrame, group_col: str, cols: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    """按 symbol 分组，添加滞后特征"""
    df = df.sort_values([group_col, "datetime"]).copy()
    for L in lags:
        for c in cols:
            df[f"{c}_lag{L}"] = df.groupby(group_col)[c].shift(L)
    return df


def _add_group_rollings(df: pd.DataFrame, group_col: str, cols: Iterable[str], windows: Iterable[int]) -> pd.DataFrame:
    """按 symbol 分组，添加滚动均值和标准差特征"""
    df = df.sort_values([group_col, "datetime"]).copy()
    for w in windows:
        win = f"roll{w}"
        for c in cols:
            g = df.groupby(group_col)[c]
            df[f"{c}_{win}_mean"] = g.transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).mean())
            df[f"{c}_{win}_std"]  = g.transform(lambda s: s.rolling(w, min_periods=max(1, w//2)).std())
    return df


def _time_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按时间顺序划分训练/测试集"""
    df = df.sort_values(["symbol", "datetime"]).copy()
    n = len(df)
    cut = int(n * float(train_frac))
    return df.iloc[:cut], df.iloc[cut:]


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """计算模型评估指标"""
    err = y_true - y_pred
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs(err / np.clip(np.abs(y_true), 1e-12, None)))) * 100.0
    r2 = float(r2_score(y_true, y_pred))
    return {"R2": r2, "MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE_%": mape}


# ------------------------- 主流水线 -------------------------

@dataclass
class SafeRandomForestPipeline:
    features: List[str]                   # 输入特征列（原始）
    target: str = "close"                 # 预测目标（默认收盘价）
    group_col: str = "symbol"            # 分组列（标的代码）
    lags: Tuple[int, ...] = (1,)          # 滞后阶数
    add_rolling: bool = True              # 是否生成滚动统计特征
    rolling_windows: Tuple[int, ...] = (5, 20)  # 滚动窗口
    clip_outliers: bool = True            # 是否缩尾处理极端值
    outlier_q: Tuple[float, float] = (0.01, 0.99)
    log_volume: bool = True               # 是否对成交量做 log1p
    log_amount: bool = True               # 是否对成交额做 log1p
    rf_n_estimators: int = 300            # 随机森林树数
    rf_max_depth: Optional[int] = 12      # 最大深度
    rf_min_samples_leaf: int = 3          # 叶子最小样本数
    rf_n_jobs: int = -1                   # 并行线程数
    random_state: int = 42                # 随机种子
    use_target_lag_as_feature: bool = True

    # 训练后填充
    model_: Optional[RandomForestRegressor] = None
    train_idx_: Optional[pd.Index] = None
    test_idx_: Optional[pd.Index] = None
    used_features_: Optional[List[str]] = None
    metrics_: Optional[Dict[str, float]] = None
    baseline_metrics_: Optional[Dict[str, float]] = None

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理：检查列、缩尾、生成滞后和滚动特征、log 变换"""
        _ensure_columns(df, ["datetime", self.group_col, self.target])

        # 优先使用 volume_std
        src_features = self.features.copy()
        if "volume_std" not in df.columns and "volume_std" in src_features and "volume" in df.columns:
            src_features[src_features.index("volume_std")] = "volume"
        _ensure_columns(df, src_features)

        work = df.sort_values([self.group_col, "datetime"]).copy()

        # 缩尾
        if self.clip_outliers:
            cols_to_clip = [c for c in ["volume_std", "volume", "amount"] if c in work.columns]
            if cols_to_clip:
                work = _winsorize(work, cols_to_clip, self.outlier_q[0], self.outlier_q[1])

        # 滞后特征
        work = _add_group_lags(work, self.group_col, src_features, self.lags)

        # 滚动特征
        if self.add_rolling:
            roll_cols = [c for c in ["close", "volume_std", "volume", "amount"] if c in work.columns]
            if roll_cols:
                work = _add_group_rollings(work, self.group_col, roll_cols, self.rolling_windows)

        # log 变换
        for L in self.lags:
            if self.log_volume:
                for cand in (f"volume_std_lag{L}", f"volume_lag{L}"):
                    if cand in work.columns:
                        work[f"{cand}_log"] = np.log1p(np.clip(work[cand].astype(float), 0, None))
            if self.log_amount:
                cand = f"amount_lag{L}"
                if cand in work.columns:
                    work[f"{cand}_log"] = np.log1p(np.clip(work[cand].astype(float), 0, None))
        work[f"{self.target}_lag1"] = (
            work.groupby(self.group_col)[self.target].shift(1)
        )
        return work

    def _collect_feature_names(self, df: pd.DataFrame) -> List[str]:
        """收集实际可用的特征列（滞后、滚动、log）"""
        feats: List[str] = []
        for L in self.lags:
            for f in self.features:
                f_eff = f
                if f == "volume_std" and f not in df.columns and "volume" in df.columns:
                    f_eff = "volume"
                name = f"{f_eff}_lag{L}"
                if name in df.columns:
                    feats.append(name)
        if self.add_rolling:
            for w in self.rolling_windows:
                for base in ["close", "volume_std", "volume", "amount"]:
                    for stat in ("mean", "std"):
                        cand = f"{base}_roll{w}_{stat}"
                        if cand in df.columns:
                            feats.append(cand)
        for L in self.lags:
            for cand in (f"volume_std_lag{L}_log", f"volume_lag{L}_log", f"amount_lag{L}_log"):
                if cand in df.columns:
                    feats.append(cand)
        # 去重
        uniq, seen = [], set()
        for f in feats:
            if f not in seen:
                uniq.append(f)
                seen.add(f)
        if self.use_target_lag_as_feature:
            cand = f"{self.target}_lag1"
            if cand in df.columns:
                feats.append(cand)
        return uniq

    def fit(self, df: pd.DataFrame, train_frac: float = 0.8) -> "SafeRandomForestPipeline":
        """训练随机森林模型，并在测试集上评估，同时计算 baseline"""
        work = self._prepare(df)
        used_features = self._collect_feature_names(work)
        work = work.dropna(subset=used_features + [self.target])

        train, test = _time_split(work, train_frac)
        self.train_idx_, self.test_idx_ = train.index, test.index

        X_train = train[used_features].to_numpy(dtype=float)
        y_train = train[self.target].to_numpy(dtype=float)
        X_test  = test[used_features].to_numpy(dtype=float)
        y_test  = test[self.target].to_numpy(dtype=float)

        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            n_jobs=self.rf_n_jobs,
            random_state=self.random_state,
        )
        rf.fit(X_train, y_train)
        self.model_ = rf
        self.used_features_ = used_features

        y_pred = rf.predict(X_test)
        self.metrics_ = _evaluate(y_test, y_pred)

        # baseline: 使用昨日 close 预测今日 close
        if f"{self.target}_lag1" in test.columns:
            yb = test[f"{self.target}_lag1"].to_numpy(dtype=float)
            self.baseline_metrics_ = _evaluate(y_test, yb)
        else:
            self.baseline_metrics_ = None

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """预测给定数据的目标值"""
        if self.model_ is None:
            raise RuntimeError("请先调用 fit() 训练模型")
        work = self._prepare(df)
        used_features = self._collect_feature_names(work)
        work = work.dropna(subset=used_features)
        X = work[used_features].to_numpy(dtype=float)
        y_pred = self.model_.predict(X)
        return pd.Series(y_pred, index=work.index, name=f"{self.target}_pred")

    def predict_on_index(self, full_df: pd.DataFrame, target_index: pd.Index) -> pd.Series:
        """
        用 full_df 构造滞后/滚动特征，然后只对 target_index 对应的行输出预测。
        这样既能保证有历史上下文，又能精确输出你想要的行。
        """
        if self.model_ is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        work = self._prepare(full_df)
        used_features = self._collect_feature_names(work)

        # 仅保留目标索引与其可用特征的交集
        valid = work.dropna(subset=used_features)
        subset = valid.loc[valid.index.intersection(target_index)]
        if subset.empty:
            raise ValueError(
                "目标索引在构造特征后无可用样本："
                "请确保为每个 symbol 提供至少 max(lags) 条历史，"
                "以及滚动窗口所需的历史（若开启 rolling）。"
            )
        X = subset[used_features].to_numpy(dtype=float)
        y_pred = self.model_.predict(X)
        return pd.Series(y_pred, index=subset.index, name=f"{self.target}_pred")

    def feature_importances(self) -> pd.Series:
        """返回特征重要性"""
        if self.model_ is None or self.used_features_ is None:
            raise RuntimeError("模型尚未训练")
        imp = pd.Series(self.model_.feature_importances_, index=self.used_features_, name="importance")
        return imp.sort_values(ascending=False)

    def report(self) -> Dict[str, Dict[str, float]]:
        """返回随机森林与 baseline 的评估结果"""
        return {
            "random_forest": self.metrics_ or {},
            "naive_baseline": self.baseline_metrics_ or {},
        }
