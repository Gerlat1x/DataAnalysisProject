from __future__ import annotations
from typing import Any, Dict, Callable

MODEL_REGISTRY: dict[str, Callable] = {}


def create_model(cfg: Dict[str, Any]):
    """
    根据配置创建模型实例。
    """
    cfg = cfg.copy()
    model_type = cfg.pop("type")
    try:
        model_cls = MODEL_REGISTRY[model_type]
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")
    return model_cls(**cfg)


from . import baseline
from . import linear_regression
from . import momentum
