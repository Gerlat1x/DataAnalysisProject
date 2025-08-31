import pandas as pd
from models.momentum import MomentumModel

# 构造测试数据
data = [
    # SYM1
    ("2022-01-03", "SYM1", 10.0),
    ("2022-01-04", "SYM1", 11.0),
    ("2022-01-05", "SYM1", 12.0),
    ("2022-01-06", "SYM1", 13.0),
    ("2022-01-07", "SYM1", 15.0),
    ("2022-01-10", "SYM1", 14.0),
    # SYM2
    ("2022-01-03", "SYM2", 20.0),
    ("2022-01-04", "SYM2", 22.0),
    ("2022-01-05", "SYM2", 21.0),
    ("2022-01-06", "SYM2", 23.0),
    ("2022-01-07", "SYM2", 24.0),
    ("2022-01-10", "SYM2", 25.0),
]
panel = pd.DataFrame(data, columns=["datetime", "symbol", "close"])
panel["datetime"] = pd.to_datetime(panel["datetime"])

model = MomentumModel(window=5)
score = model.score(panel)

print(score)  # 期望：每个symbol仅1个值
