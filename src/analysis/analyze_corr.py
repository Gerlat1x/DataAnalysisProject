import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("../../data/processed/binary_dataset.parquet")

cols = ["f_rsi_14", "f_mom_5", "f_mom_10", "f_mom_20", "f_ret_mean_w"]
sub = df[cols].dropna()

# 3. 计算相关系数矩阵
corr = sub.corr()

print("=== Correlation Matrix ===")
print(corr)

# 4. 画热力图
plt.figure(figsize=(6,5))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
plt.yticks(range(len(cols)), cols)

plt.title("Correlation of RSI vs Momentum Features")
plt.tight_layout()
plt.show()
