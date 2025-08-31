import pandas as pd

df = pd.read_parquet("../data/processed/panel.parquet")

ratio = (df['amount'] / (df['close'] * df['volume'].replace(0, pd.NA))).groupby(df['symbol']).median()

# 转成数值型，忽略非数值
ratio = pd.to_numeric(ratio, errors="coerce")

# 生成单位映射表（1 或 100）
unit_map = ratio.round().astype("Int64").to_dict()

# 映射到原表
df['vol_unit'] = df['symbol'].map(unit_map).fillna(1).astype(int)

# 统一换算为股
df['volume_std'] = df['volume'] * df['vol_unit']

print(ratio.value_counts().head(10))

# print("行数/列数:", df.shape)
# print("日期范围:", df['datetime'].min(), "→", df['datetime'].max())
# print("标的数量:", df['symbol'].nunique())
# need = ["datetime", "symbol", "open", "high", "low", "close", "volume", "amount", "pre_close"]
# print("必要列是否齐全:", set(need).issubset(df.columns))
#
# # 简单缺失率
# na = df[need].isna().mean().sort_values(ascending=False)
# print("缺失率：\n", na.head(10))
#
# # 是否按 (datetime, symbol) 唯一
# dup = df.duplicated(subset=["datetime", "symbol"]).sum()
# print("重复键数量:", dup)
#
# numeric_cols = [c for c in need if c not in ["datetime", "symbol"]]
# desc = df[numeric_cols].agg(["mean", "std", "min", "max"])
# print("主要数值列描述统计：\n", desc)
#
# # 非负检查
# non_negative_cols = ["open", "high", "low", "close", "pre_close", "volume", "amount"]
# for col in non_negative_cols:
#     negative = (df[col] < 0).sum()
#     if negative:
#         print(f"{col} 存在 {negative} 个负值")
#     else:
#         print(f"{col} 无负值")
#
# # 交易量是否过大（例如超过 1e12）
# max_vol = df["volume"].max()
# print("最大成交量:", max_vol)
# if max_vol > 1e12:
#     print("警告：存在异常大的成交量")
#
# # 简单异常点检测：以5倍标准差作为阈值
# z_scores = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
# outliers = (z_scores.abs() > 5).any(axis=1).sum()
# print("明显异常点数量:", outliers)
#
# # 类型与排序
# assert pd.api.types.is_datetime64_any_dtype(df['datetime']), "datetime 不是时间类型"
# df = df.sort_values(['symbol', 'datetime']).reset_index(drop=True)
# print("按 (symbol, datetime) 已排序")
#
# # high/low 包含 open/close
# bad_hilo = df[(df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
#               (df['low'] > df[['open', 'close', 'high']].min(axis=1))]
# print("OHLC 违反边界的行数:", len(bad_hilo))
#
# # 逐 symbol，用昨日收盘对齐今日 pre_close
# df['prev_close'] = df.groupby('symbol')['close'].shift(1)
# tol = 1e-6  # 或者用相对误差
# bad_preclose = df[((df['pre_close'] - df['prev_close']).abs() > tol) & df['prev_close'].notna()]
# print("pre_close 未对齐的行数:", len(bad_preclose))
#
#
# # 估计 volume 的单位（股=1；手=100）
# # 用中位数估计 ratio = amount / (close * volume * k) 接近 1
# def infer_vol_unit(sub):
#     # 只用正值样本估计
#     s = sub[(sub['amount'] > 0) & (sub['close'] > 0) & (sub['volume'] > 0)].copy()
#     if len(s) < 50:
#         return 1  # 样本太少默认股
#     r1 = (s['amount'] / (s['close'] * s['volume'])).median()
#     r100 = (s['amount'] / (s['close'] * s['volume'] * 100)).median()
#     return 1 if abs(r1 - 1) < abs(r100 - 1) else 100
#
#
# unit_map = df.groupby('symbol').apply(infer_vol_unit).to_dict()
# df['vol_unit'] = df['symbol'].map(unit_map).astype(int)
#
# # 用推断单位做一致性检查（允许 30% 相对误差的宽松阈值，避免撮合细节/加权价差）
# rel_tol = 0.3
# est_amount = df['close'] * df['volume'] * df['vol_unit']
# bad_amount = df[(df['amount'] > 0) & (est_amount > 0) &
#                 (((df['amount'] - est_amount).abs() / est_amount) > rel_tol)]
# print("成交额与价格*量不一致（超阈值）的行数:", len(bad_amount))
#
# zero_vol_jump = df[(df['volume'] == 0) & (df['close'] > 0) & (df['pre_close'] > 0) &
#                    ((df['close'] / df['pre_close'] - 1).abs() > 1e-6)]
# print("零成交量但价格变动的行数:", len(zero_vol_jump))
#
#
# # 逐标的用 MAD 检测体量异常
# def mad_flags(s, k=5.0):
#     med = s.median()
#     mad = (s - med).abs().median()
#     if mad == 0:
#         return pd.Series(False, index=s.index)
#     z = 0.6745 * (s - med) / mad
#     return z.abs() > k
#
#
# vol_outlier = df.groupby('symbol')['volume'].transform(lambda x: mad_flags(x, k=7.0))
# print("成交量稳健异常点数:", vol_outlier.sum())
#
#
# # 假设你用交易日历（已有 df 的日期集合），逐 symbol 检查缺口
# def missing_dates(gr):
#     full = pd.DatetimeIndex(sorted(gr['datetime'].unique()))
#     # 如果有官方交易日历更好；这里用各自出现过的日期做近似
#     gaps = full[1:][(full[1:] - full[:-1]) > pd.Timedelta(days=3)]  # >3天的缺口（含周末跨长假）
#     return pd.Series({"gaps_count": len(gaps)})
#
#
# gap_stats = df.groupby('symbol').apply(missing_dates)
# print("出现长缺口的标的数量:", (gap_stats['gaps_count'] > 0).sum())
