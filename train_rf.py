import numpy as np

from src.models.random_forest import SafeRandomForestPipeline
import pandas as pd

"""
更稳健的训练脚本（train_rf.py）
- 解决预测阶段缺少历史上下文导致的 "0 samples" 问题：
  直接对全量 df 进行 predict，再在目标索引上取子集。
- 同时输出 随机森林 vs Naive baseline 的对比表。

使用：python train_rf.py
"""

# 1) 读取数据（按需修改路径）
df = pd.read_parquet("data/processed/panel.parquet")

if not pd.api.types.is_datetime64_dtype(df["datetime"]):
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.sort_values(["symbol", "datetime"]).reset_index(drop=True)

pre = df["pre_close"].replace(0, np.nan)
df["ret1"] = df["close"] / pre - 1.0  # 1 日收益率
df = df.dropna(subset=["ret1"])  # 丢弃 pre_close=0 的行

# 2) 选择特征列（若没有 volume_std 会自动在 pipeline 内回退到 volume）
features = ["open", "high", "low", "volume_std", "amount"]

# 3) 初始化流水线（保持与 random_forest_pipeline.py 中示例一致）


pipeline = SafeRandomForestPipeline(
    features=features,
    target="ret1",
    lags=(1,),  # 先用 lag1，后续再加 (1,2,5)
    add_rolling=True,  # 滚动统计
    rolling_windows=(5, 20),
    clip_outliers=True,  # 缩尾极端值
    log_volume=True,
    log_amount=True,
    # 更稳的随机森林参数（可按需调整）
    rf_n_estimators=600,
    rf_max_depth=None,  # None = 不限制深度（配合 min_samples_leaf）
    rf_min_samples_leaf=5,
)

# 4) 训练（内部按时间顺序 80/20 切分）
pipeline.fit(df, train_frac=0.8)

# 5) 指标输出（随机森林 & baseline）
print("==== 评估结果（字典） ====")
print(pipeline.report())

# 6) 组装对比表，便于一眼比较
rf = pipeline.report().get("random_forest", {})
bl = pipeline.report().get("naive_baseline", {})
metrics = ["R2", "RMSE", "MAE", "MAPE_%", "MSE"]
compare_df = pd.DataFrame(
    [{"metric": m, "random_forest": rf.get(m), "naive_baseline": bl.get(m)} for m in metrics]
).set_index("metric")
print("\n==== 随机森林 vs Naive baseline ====")
print(compare_df)

# 6) 特征重要性（Top 20）
print("\n==== 特征重要性（Top 20） ====")
print(pipeline.feature_importances().head(20))

# 7) 预测：全量预测，再映射到原始最后 20 行索引
pred_all = pipeline.predict(df)
last_idx = df.tail(20).index
pred_tail20 = pred_all.loc[pred_all.index.intersection(last_idx)]
print("\n==== 预测结果（映射到原始最后20行索引） ====")
print(pred_tail20)
