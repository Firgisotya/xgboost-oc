from flask import render_template, request, redirect, session, flash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from models.hst_best_param_model import HstBestParamModel
from helpers.util_helper import generate_random_string
from helpers.preprocessing_helper import PreprocessingHelper
# from helpers.rscv_helper import RandomizedSaerchHelper

# param_grid = {
#         "objective": ["reg:squarederror", "reg:linear"],
#         "booster": ["gbtree", "gblinear"],
#         "learning_rate": np.linspace(0.01, 0.2, 20).tolist(),
#         "max_depth": range(3, 10),
#         "n_estimators": range(100, 1000, 100),
#         "gamma": np.linspace(0.01, 3, 30).tolist(),
#         "colsample_bytree": np.linspace(0.5, 1, 6).tolist(),
#         "subsample": np.linspace(0.5, 1, 6).tolist(),
#         "reg_alpha": np.linspace(0.2, 1, 9).tolist(),
#         "reg_lambda": range(1, 5),
#         "min_child_weight": range(1, 25),
# }

param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7 ,10],
    "n_estimators": [300, 400, 500, 600],
    "min_child_weight": [1, 2, 3],
    "colsample_bytree": [0.5, 0.7],
    "subsample": [0.5, 0.7],
    "gamma": [0, 1, 2],
}


class TrainModelController:
    def __init__(self):
        self.hstBestParam = HstBestParamModel()
        self.preprocessing_helper = PreprocessingHelper()
        # self.best_params = RandomizedSaerchHelper()
    
    def index(self):
        try:
            mape = None
            optimized_params = None

            if request.method == 'POST':
                try:
                    X_train, X_test, y_train, y_test = self.preprocessing_helper.load_dataset()

                    optimized_params = GridSearchCV(
                        xgb.XGBRegressor(),
                        param_grid,
                        cv=3,
                        # n_iter=10,
                        scoring='neg_mean_absolute_percentage_error',
                        verbose=0,
                        n_jobs=-1
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        optimized_params.fit(
                            X_train,
                            y_train,
                            early_stopping_rounds=10,
                            eval_set=[(X_test, y_test)],
                            verbose=False
                        )

                
                    if optimized_params is not None:
                        best_params = optimized_params.best_params_
                        print(f'Best params: {best_params}')

                        model = xgb.XGBRegressor(
                            # objective=best_params["objective"],
                            # booster=best_params["booster"],
                            eval_metric="mape",
                            learning_rate=best_params["learning_rate"],
                            gamma=best_params["gamma"],
                            max_depth=best_params["max_depth"],
                            n_estimators=best_params["n_estimators"],
                            colsample_bytree=best_params['colsample_bytree'],
                            subsample=best_params['subsample'],
                            # reg_alpha=best_params['reg_alpha'],
                            # reg_lambda=best_params['reg_lambda'],
                            min_child_weight=best_params['min_child_weight']
                        )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mape = mean_absolute_percentage_error(y_test, y_pred)
                        mape = round(mape, 3)
                        print(f'MAPE: {mape}')

                        # self.hstBestParam.create({
                        #     'id': generate_random_string(5),
                        #     'objective': best_params['objective'],
                        #     'booster': best_params['booster'],
                        #     'learning_rate': best_params['learning_rate'],
                        #     'max_depth': best_params['max_depth'],
                        #     'n_estimators': best_params['n_estimators'],
                        #     'gamma': best_params['gamma'],
                        #     'colsample_bytree': best_params['colsample_bytree'],
                        #     'subsample': best_params['subsample'],
                        #     'reg_alpha': best_params['reg_alpha'],
                        #     'reg_lambda': best_params['reg_lambda'],
                        #     'min_child_weight': best_params['min_child_weight'],
                        #     'mape': mape
                        # })

                    else:
                        flash('Failed to find best parameters.')
                        return redirect('/latih-model')

                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html', optimized_params=optimized_params, mape=mape)

        except Exception as e:
            print(f"Error in index method: {e}")
            flash('An unexpected error occurred.')
            return redirect('/latih-model')

