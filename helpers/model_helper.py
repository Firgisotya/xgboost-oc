import os
import joblib
import numpy as np
import xgboost as xgb

def xgboost_model(solution):
    learning_rate, min_split_loss, max_depth, n_estimators = solution

    # learning_rate
    learning_rate = max(0.1, min(0.001, float(learning_rate)))

    # max_depth [7, 10, 15, 20]
    max_depth = max(20, min(2, int(max_depth)))

    # n_estimators [300, 400, 500, 600]
    n_estimators = max(1000, min(100, int(n_estimators)))

    # min_split_loss [1, 2, 3]
    min_split_loss = max(3, min(0.01, float(min_split_loss)))

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        eval_metric="mae",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_split_loss=min_split_loss
    )

    return model