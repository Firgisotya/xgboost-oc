from flask import render_template, request, redirect, session, flash
from helper import render, view
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error
from models.hst_best_param_model import HstBestParamModel
from helpers.util_helper import generate_random_string
from helpers.preprocessing_helper import PreprocessingHelper

class TrainModelController:
    def __init__(self):
        self.hstBestParam = HstBestParamModel()
        self.preprocessing_helper = PreprocessingHelper()
    
    def index(self):
        try:
            booster_options = ['gbtree', 'gblinear'],
            learning_rate_options = [0.1, 0.01, 0.001, 0.5]
            max_depth_options = [2, 3, 4, 5, 6, 7, 10, 12, 15, 20]
            n_estimators_options = [100, 250, 300, 400, 500, 600, 1000]
            min_split_loss_options = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 2, 3],
            colsample_bytree_options = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            subsample_options = [0.2, 0.4, 0.5, 0.6, 0.7],
            reg_alpha_options = [0, 0.2, 0.5, 1],
            reg_lambda_options = [1, 1.5, 2, 3, 4.5, 5],
            min_child_weight_options = [1, 3, 5, 7, 10, 15, 20, 25],

            mape = None
            optimized_params = None

            if request.method == 'POST':
                try:

                    param_distributions = {
                        'booster': booster_options,
                        'learning_rate': learning_rate_options,
                        'max_depth': max_depth_options,
                        'n_estimators': n_estimators_options,
                        'min_split_loss': min_split_loss_options,
                        'colsample_bytree': colsample_bytree_options,
                        'subsample': subsample_options,
                        'reg_alpha': reg_alpha_options,
                        'reg_lambda': reg_lambda_options,
                        'min_child_weight': min_child_weight_options
                    }

                    X_train, X_test, y_train, y_test = self.preprocessing_helper.load_dataset()
                    optimized_params = RandomizedSearchCV(
                        xgb.XGBRegressor(),
                        param_distributions,
                        n_iter=10,
                        n_jobs=-1,
                        cv=5,
                        verbose=3
                    )
                    optimized_params.fit(
                        X_train, 
                        y_train,
                        early_stopping_rounds=10,
                        eval_set=[(X_test, y_test)],
                        verbose=False
                        )
                    
                    print(f'Best params: {optimized_params.best_params_}')

                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        booster=optimized_params.best_params_['booster'],
                        eval_metric='mae',
                        learning_rate=optimized_params.best_params_['learning_rate'],
                        max_depth=optimized_params.best_params_['max_depth'],
                        n_estimators=optimized_params.best_params_['n_estimators'],
                        min_split_loss=optimized_params.best_params_['min_split_loss'],
                        colsample_bytree=optimized_params.best_params_['colsample_bytree'],
                        subsample=optimized_params.best_params_['subsample'],
                        reg_alpha=optimized_params.best_params_['reg_alpha'],
                        reg_lambda=optimized_params.best_params_['reg_lambda'],
                        min_child_weight=optimized_params.best_params_['min_child_weight']
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    mape = round(mape, 3)
                    print(f'MAPE: {mape}')

                    # self.hstBestParam.create({
                    #     'id': generate_random_string(5),
                    #     'learning_rate': optimized_params.best_params_['learning_rate'],
                    #     'max_depth': optimized_params.best_params_['max_depth'],
                    #     'n_estimators': optimized_params.best_params_['n_estimators'],
                    #     'min_split_loss': optimized_params.best_params_['min_split_loss'],
                    #     'mape': mape
                    # })

                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html', optimized_params=optimized_params, mape=mape)

        except Exception as e:
            print(f"Error in index method: {e}")

