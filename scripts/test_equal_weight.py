import pandas as pd
from src.decisions.equal_weight import TopKEqualWeightDecision


def main():
    # 构造一个虚拟的得分序列（假设 5 只股票）
    scores = pd.Series(
        {"AAPL": 0.9, "MSFT": 0.8, "TSLA": 0.95, "AMZN": 0.6, "META": 0.7},
        name="score"
    )
    print("原始分数:")
    print(scores, "\n")

    # 测试 Top-3 等权
    decision = TopKEqualWeightDecision(k=3)
    weights = decision.allocate(scores)
    print("Top-3 等权分配结果:")
    print(weights, "\n")

    # 测试空输入
    weights_empty = decision.allocate(None)
    print("空输入分配结果:")
    print(weights_empty, "\n")

    # 测试有 NaN 的情况
    scores_with_nan = scores.copy()
    scores_with_nan["TSLA"] = float("nan")
    weights_nan = decision.allocate(scores_with_nan)
    print("含 NaN 输入分配结果:")
    print(weights_nan, "\n")


if __name__ == "__main__":
    main()
