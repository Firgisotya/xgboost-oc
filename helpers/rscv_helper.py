import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import warnings

param_grid = {
        "objective": ["reg:squarederror", "reg:linear"],
        "booster": ["gbtree", "gblinear"],
        "learning_rate": [0.1, 0.01, 0.001, 0.5],
        "max_depth": [2, 3, 4, 5, 6, 7, 10, 12, 15, 20],
        "n_estimators": [100, 250, 300, 400, 500, 600, 1000],
        "gamma": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 3],
        "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
        "reg_alpha": [0, 0.2, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3, 4.5, 5],
        "min_child_weight": [1, 3, 5, 7, 10, 15, 20, 25],
}

class RandomizedSaerchHelper:
    def __init__(self):
        pass

    def find_best_params(X_train, y_train, X_test, y_test):
        try:
            optimized_params = RandomizedSearchCV(
                xgb.XGBRegressor(),
                param_grid,
                cv=5,
                n_iter=10,
                scoring='neg_mean_absolute_error',
                verbose=3,
                n_jobs=12
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                optimized_params.fit(
                    X_train,
                    y_train,
                    early_stopping_rounds=10,
                    eval_set=[(X_test, y_test)],
                    verbose=1
                )

            return optimized_params.best_params_
        except Exception as e:
            print(f"Error during hyperparameter tuning: {e}")
            return None
