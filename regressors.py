import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

# -------------------------
# Torch MLP 回归器（可反传）
# -------------------------
import torch
import torch.nn as nn


class TorchMLPRegressor(nn.Module):
    """输入: (B, D) 输出: (B,)"""
    def __init__(self, in_dim: int, hidden: str = "256,128", dropout: float = 0.1):
        super().__init__()
        hs = [int(x) for x in hidden.split(",") if x.strip()]
        layers = []
        d = in_dim
        for h in hs:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# -------------------------
# Sklearn 回归器（不可反传）
# -------------------------
def _require_sklearn():
    try:
        import sklearn  # noqa
        import joblib   # noqa
    except Exception as e:
        raise RuntimeError("需要安装 scikit-learn 与 joblib：pip install scikit-learn joblib") from e


@dataclass
class SklearnRegressorBundle:
    model_name: str
    model: Any
    scaler: Any
    y_mean: float
    y_std: float
    label_zscore: bool

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X) if self.scaler is not None else X
        y_pred = self.model.predict(Xs)
        if self.label_zscore:
            y_pred = y_pred * self.y_std + self.y_mean
        return y_pred


def make_sklearn_bundle(
    model_name: str,
    label_zscore: bool,
    y_mean: float,
    y_std: float,
    params: Optional[Dict[str, Any]] = None,
) -> SklearnRegressorBundle:
    _require_sklearn()
    params = params or {}

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    # 常用回归器
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

    name = model_name.lower()

    # 哪些模型建议做 StandardScaler
    need_scaler = name in {"ridge", "lasso", "elasticnet", "svr", "knn", "sgd"}

    scaler = StandardScaler() if need_scaler else None

    if name == "ridge":
        model = Ridge(alpha=float(params.get("alpha", 1.0)), random_state=int(params.get("seed", 0)))
    elif name == "lasso":
        model = Lasso(alpha=float(params.get("alpha", 1e-3)), max_iter=int(params.get("max_iter", 5000)), random_state=int(params.get("seed", 0)))
    elif name == "elasticnet":
        model = ElasticNet(alpha=float(params.get("alpha", 1e-3)), l1_ratio=float(params.get("l1_ratio", 0.5)),
                           max_iter=int(params.get("max_iter", 5000)), random_state=int(params.get("seed", 0)))
    elif name == "svr":
        model = SVR(C=float(params.get("C", 10.0)), epsilon=float(params.get("epsilon", 0.1)), kernel=str(params.get("kernel", "rbf")))
    elif name == "knn":
        model = KNeighborsRegressor(n_neighbors=int(params.get("k", 8)), weights=str(params.get("weights", "distance")))
    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 400)),
            max_depth=params.get("max_depth", None),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            random_state=int(params.get("seed", 0)),
            n_jobs=int(params.get("n_jobs", -1)),
        )
    elif name == "gbrt":
        model = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            learning_rate=float(params.get("lr", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=int(params.get("seed", 0)),
        )
    elif name == "extratrees":
        model = ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=params.get("max_depth", None),
            random_state=int(params.get("seed", 0)),
            n_jobs=int(params.get("n_jobs", -1)),
        )
    elif name == "sgd":
        # 支持 partial_fit（可做“epoch”）
        model = SGDRegressor(
            loss=str(params.get("loss", "huber")),
            alpha=float(params.get("alpha", 1e-4)),
            learning_rate=str(params.get("learning_rate", "invscaling")),
            eta0=float(params.get("eta0", 1e-3)),
            random_state=int(params.get("seed", 0)),
        )
    else:
        raise ValueError(f"Unknown reg_model: {model_name}")

    # 用 Pipeline 统一处理 scaler
    if scaler is not None:
        pipe = Pipeline([("scaler", scaler), ("model", model)])
        bundle = SklearnRegressorBundle(name, pipe, None, y_mean, y_std, label_zscore)
    else:
        bundle = SklearnRegressorBundle(name, model, None, y_mean, y_std, label_zscore)

    return bundle


def sklearn_fit(bundle: SklearnRegressorBundle, X: np.ndarray, y: np.ndarray):
    # y 若 label_zscore=True，这里传入的是归一化后的 y
    bundle.model.fit(X, y)


def sklearn_partial_fit(bundle: SklearnRegressorBundle, X: np.ndarray, y: np.ndarray):
    # 只对 SGDRegressor 这种支持 partial_fit 的模型有效（在 Pipeline 里要取到末端模型）
    mdl = bundle.model
    if hasattr(mdl, "partial_fit"):
        mdl.partial_fit(X, y)
        return

    # Pipeline 情况
    if hasattr(mdl, "named_steps") and "model" in mdl.named_steps:
        # scaler 先 fit/transform
        if "scaler" in mdl.named_steps:
            sc = mdl.named_steps["scaler"]
            # 第一次调用应先 fit；这里外部保证先 fit 一次 scaler
            Xs = sc.transform(X)
        else:
            Xs = X
        base = mdl.named_steps["model"]
        if not hasattr(base, "partial_fit"):
            raise RuntimeError("该模型不支持 partial_fit")
        base.partial_fit(Xs, y)
        return

    raise RuntimeError("该模型不支持 partial_fit")


def sklearn_save(bundle: SklearnRegressorBundle, path: str):
    _require_sklearn()
    import joblib
    joblib.dump(bundle, path)


def sklearn_load(path: str) -> SklearnRegressorBundle:
    _require_sklearn()
    import joblib
    return joblib.load(path)
