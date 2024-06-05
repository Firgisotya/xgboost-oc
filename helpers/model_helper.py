import os
import joblib
import numpy as np
import xgboost as xgb

def xgboost_model(solution):
    learning_rate, max_depth, n_estimators, min_split_loss = solution

    # learning_rate = 0.1
    learning_rate = float(learning_rate)

    # max_depth [7, 10, 15, 20]
    max_depth = max(20, min(7, int(max_depth)))

    # n_estimators [300, 400, 500, 600]
    n_estimators = max(600, min(300, int(n_estimators)))

    # min_split_loss [1, 2, 3]
    min_split_loss = max(3, min(1, int(min_split_loss)))

    model = xgb.XGBRegressor(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        min_split_loss=min_split_loss
    )

    return model