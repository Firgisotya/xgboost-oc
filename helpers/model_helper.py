import os
import joblib
import numpy as np
import xgboost as xgb

def xgboost_model(solution):
    objective, booster, learning_rate, max_depth, n_estimators, gamma, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight = solution

    objective = str(objective)
    booster = str(booster)
    learning_rate = float(learning_rate)
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    gamma = float(gamma)
    subsample = float(subsample)
    colsample_bytree = float(colsample_bytree)
    reg_alpha = float(reg_alpha)
    reg_lambda = float(reg_lambda)
    min_child_weight = int(min_child_weight)
    

    model = xgb.XGBRegressor(
        objective=objective,
        booster=booster,
        eval_metric="mape",
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight
    )

    return model