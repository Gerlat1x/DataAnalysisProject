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

# 2) 选择特征列（若没有 volume_std 会自动在 pipeline 内回退到 volume）
features = ["open", "high", "low", "volume_std", "amount"]

# 3) 初始化流水线（保持与 random_forest_pipeline.py 中示例一致）

df["ret1"] = df["close"] / df["pre_close"] - 1.0  # 1 日收益率

pipeline = SafeRandomForestPipeline(
    features=["open","high","low","volume_std","amount"],
    target="ret1",
    lags=(1,),                 # 使用 1 日滞后
    add_rolling=True,          # 添加滚动统计特征
    rolling_windows=(5, 20),   # 5/20 日均值与波动
    clip_outliers=True,        # 缩尾极端值
    log_volume=True,
    log_amount=True,
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
rows = []
for m in metrics:
    rows.append({"metric": m,
                 "random_forest": rf.get(m, None),
                 "naive_baseline": bl.get(m, None)})
compare_df = pd.DataFrame(rows).set_index("metric")
print("\n==== 随机森林 vs Naive baseline ====")
print(compare_df)

# 7) 查看特征重要性（取前 20 个）
print("\n==== 特征重要性（Top 20） ====")
print(pipeline.feature_importances().head(20))

# 8) 预测：对全量 df 预测，再对目标索引取子集
#    这样能保证滞后/滚动特征有充足历史，不会出现 0 样本。
pred_all = pipeline.predict(df)

# 举例：取原始数据最后 20 行对应的预测结果（可能跨多个 symbol）
last_idx = df.tail(20).index
pred_tail20 = pred_all.loc[pred_all.index.intersection(last_idx)]
print("\n==== 预测结果（映射到原始最后20行索引） ====")
print(pred_tail20)
