import pandas as pd
df = pd.read_parquet("../data/processed/panel.parquet")

print("行数/列数:", df.shape)
print("日期范围:", df['datetime'].min(), "→", df['datetime'].max())
print("标的数量:", df['symbol'].nunique())
need = ["datetime","symbol","open","high","low","close","volume","amount","pre_close"]
print("必要列是否齐全:", set(need).issubset(df.columns))

# 简单缺失率
na = df[need].isna().mean().sort_values(ascending=False)
print("缺失率：\n", na.head(10))

# 是否按 (datetime, symbol) 唯一
dup = df.duplicated(subset=["datetime","symbol"]).sum()
print("重复键数量:", dup)