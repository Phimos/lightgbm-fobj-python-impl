import lightgbm as lgb
import numpy as np

from fobj import ObjectiveFunction
from regression import L2Loss


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
) -> lgb.Booster:
    fobj: ObjectiveFunction = L2Loss()
    train_data = lgb.Dataset(X_train, label=y_train, init_score=fobj.compute_init_score(y_train))
    valid_data = lgb.Dataset(X_valid, label=y_valid, init_score=fobj.compute_init_score(y_valid))

    params = {
        "objective": fobj.compute,
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
    }

    model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100)
    return model
