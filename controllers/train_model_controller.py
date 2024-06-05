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

class TrainModelController:
    def __init__(self):
        self.hstBestParam = HstBestParamModel()

    def load_train_dataset(self):
        file_path = os.path.join('static', 'dataset', 'data_hour_2023.csv')
        return pd.read_csv(file_path)
    
    def load_dataset(self):
        data = self.load_train_dataset()
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)
        data.dropna(inplace=True)
        data.drop(['prod_order2', 'lotno2'], axis=1, inplace=True)
        X = data.drop('value', axis=1)
        y = data['value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def index(self):
        try:
            data = {
                'learning_rate': 0.1,
                'max_depth': 15,
                'n_estimators': 600,
                'min_split_loss': 3
            }

            mape = None
            optimized_params = None

            if request.method == 'POST':
                try:
                    print(request.form)
                    data['learning_rate'] = float(request.form['learning_rate'])
                    data['max_depth'] = int(request.form['max_depth'])
                    data['n_estimators'] = int(request.form['n_estimators'])
                    data['min_split_loss'] = int(request.form['min_split_loss'])

                    param_distributions = {
                        'learning_rate': [data['learning_rate']],
                        'max_depth': [data['max_depth']],
                        'n_estimators': [data['n_estimators']],
                        'min_split_loss': [data['min_split_loss']]
                    }

                    X_train, X_test, y_train, y_test = self.load_dataset()
                    optimized_params = RandomizedSearchCV(
                        xgb.XGBRegressor(),
                        param_distributions=param_distributions,
                        n_iter=10,
                        cv=5,
                        n_jobs=-1,
                        verbose=3,
                        error_score='raise'
                    )
                    optimized_params.fit(X_train, y_train)
                    model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        booster='gbtree',
                        eval_metric='mae',
                        learning_rate=optimized_params.best_params_['learning_rate'],
                        max_depth=optimized_params.best_params_['max_depth'],
                        n_estimators=optimized_params.best_params_['n_estimators'],
                        min_split_loss=optimized_params.best_params_['min_split_loss']
                    )
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    mape = round(mape, 3)
                    print(f'MAPE: {mape}')

                    self.hstBestParam.create({
                        'learning_rate': optimized_params.best_params_['learning_rate'],
                        'max_depth': optimized_params.best_params_['max_depth'],
                        'n_estimators': optimized_params.best_params_['n_estimators'],
                        'min_split_loss': optimized_params.best_params_['min_split_loss'],
                        'mape': mape
                    })
                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html', data=data, optimized_params=optimized_params, mape=mape)

        except Exception as e:
            print(f"Error in index method: {e}")

