from flask import render_template, request, redirect, session, flash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import io
import base64
import warnings
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from models.hst_best_param_model import HstBestParamModel
from helpers.util_helper import generate_random_string
from helpers.preprocessing_helper import PreprocessingHelper
# from helpers.rscv_helper import RandomizedSaerchHelper

param_grid = {
        "objective": ["reg:squarederror", "reg:linear"],
        "booster": ["gbtree", "gblinear"],
        "learning_rate": [0.01, 0.1],
        "max_depth": [2, 3, 4, 5, 6, 7, 10, 12, 15, 20],
        "n_estimators": [100, 250, 300, 400, 500, 600],
        "gamma": [1, 2, 3],
        "colsample_bytree": [0.5, 0.7],
        "subsample": [0.5, 0.7],
        "reg_alpha": [0.2, 0.5, 1],
        "reg_lambda": [1, 1.5, 2, 3, 4.5, 5],
        "min_child_weight": [1, 3, 5, 7, 10, 15, 20, 25],
}

# param_grid = {
#     "learning_rate": [0.01, 0.1],
#     "max_depth": [3, 5, 7 ,10],
#     "n_estimators": [300, 400, 500, 600],
#     "min_child_weight": [1, 2, 3],
#     "colsample_bytree": [0.5, 0.7],
#     "subsample": [0.5, 0.7],
#     "gamma": [0, 1, 2],
# }


class TrainModelController:
    def __init__(self):
        self.hstBestParam = HstBestParamModel()
        self.preprocessing_helper = PreprocessingHelper()
        # self.best_params = RandomizedSaerchHelper()
    
    def index(self):
        try:
            mape = None
            optimized_params = None
            plot_url = None

            if request.method == 'POST':
                try:
                    X_train, X_test, y_train, y_test = self.preprocessing_helper.load_dataset()

                    model = xgb.XGBRegressor(
                            eval_metric="rmse",
                            learning_rate=0.1,
                            gamma=5,
                            reg_lambda=3,
                        )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    mape = np.round(mape, 3)
                    print("Mean Absolute Percentage Error: ", mape)

                    data_test = self.preprocessing_helper.load_data_test()
                    data_test['prediksi_reject'] = np.round(y_pred)

                    return render_template('train/index.html', data=data_test.to_dict(orient='records'), mape=mape)

                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html')

        except Exception as e:
            print(f"Error in index method: {e}")
            flash('An unexpected error occurred.')
            return redirect('/latih-model')

