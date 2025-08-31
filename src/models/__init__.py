from __future__ import annotations
from typing import Any, Dict

from .momentum import MomentumModel
from .linear_regression import LinearRegressionModel


def create_model(cfg: Dict[str, Any]):
    """
    根据配置创建模型实例。
    """
    model_type = cfg.get("type")
    if model_type == "Momentum":
        return MomentumModel(window=cfg.get("window", 5))
    if model_type == "LinearRegression":
        return LinearRegressionModel()
    raise ValueError(f"Unknown model type: {model_type}")