from flask import render_template, request, redirect, session, flash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import io
import os
import base64
import warnings
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from models.hst_best_param_model import HstBestParamModel
from helpers.util_helper import generate_random_string
from helpers.preprocessing_helper import PreprocessingHelper


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
                    X_train_3, X_test_3, y_train_3, y_test_3, X_train_6, X_test_6, y_train_6, y_test_6, X_train_12, X_test_12, y_train_12, y_test_12 = self.preprocessing_helper.load_dataset()

                    # model 3 bulan
                    model_3 = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        learning_rate=0.5,
                        gamma=2,
                        max_depth=6,
                        n_estimators=100,
                        colsample_bytree=0.4,
                        subsample=0.7,
                        reg_lambda=3,
                        min_child_weight=1,
                    )

                    model_3.fit(X_train_3, y_train_3)

                    # model 6 bulan
                    model_6 = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        learning_rate=0.5,
                        gamma=2,
                        max_depth=6,
                        n_estimators=100,
                        colsample_bytree=0.4,
                        subsample=0.7,
                        reg_lambda=3,
                        min_child_weight=1,
                    )

                    model_6.fit(X_train_6, y_train_6)

                    # model 12
                    model_12 = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        eval_metric="rmse",
                        learning_rate=0.5,
                        gamma=2,
                        max_depth=6,
                        n_estimators=100,
                        colsample_bytree=0.4,
                        subsample=0.7,
                        reg_lambda=3,
                        min_child_weight=1,
                    )

                    model_12.fit(X_train_12, y_train_12)

                    # prediksi 3 bulan
                    y_pred_3 = model_3.predict(X_test_3)
                    y_pred_3 = np.round(y_pred_3)

                    # prediksi 6 bulan
                    y_pred_6 = model_6.predict(X_test_6)
                    y_pred_6 = np.round(y_pred_6)

                    # prediksi 12 bulan
                    y_pred_12 = model_12.predict(X_test_12)
                    y_pred_12 = np.round(y_pred_12)

                    # MAPE
                    mape_3 = mean_absolute_percentage_error(y_test_3, y_pred_3)
                    mape_6 = mean_absolute_percentage_error(y_test_6, y_pred_6)
                    mape_12 = mean_absolute_percentage_error(y_test_12, y_pred_12)

                    mape_3_persen = mape_3 * 100
                    mape_6_persen = mape_6 * 100
                    mape_12_persen = mape_12 * 100

                    akurasi_3 = 100 - mape_3_persen
                    akurasi_6 = 100 - mape_6_persen
                    akurasi_12 = 100 - mape_12_persen

                    print(f"MAPE 3 Bulan: {round(mape_3, 3)}")
                    print(f"MAPE 6 Bulan: {round(mape_6, 3)}")
                    print(f"MAPE 12 Bulan: {round(mape_12, 3)}")

                    print(f"MAPE 3 Bulan (%): {mape_3_persen:.2f}%")
                    print(f"MAPE 6 Bulan (%): {mape_6_persen:.2f}%")
                    print(f"MAPE 12 Bulan (%): {mape_12_persen:.2f}%")

                    print(f"Akurasi 3 Bulan (%): {akurasi_3:.2f}%")
                    print(f"Akurasi 6 Bulan (%): {akurasi_6:.2f}%")
                    print(f"Akurasi 12 Bulan (%): {akurasi_12:.2f}%")

                    test_3.drop(columns=["Unnamed: 0", "month", "year"], inplace=True)
                    test_3["forecast"] = np.round(y_pred_3)

                    test_6.drop(columns=["Unnamed: 0", "month", "year"], inplace=True)
                    test_6["forecast"] = np.round(y_pred_6)

                    test_12.drop(columns=["Unnamed: 0", "month", "year"], inplace=True)
                    test_12["forecast"] = np.round(y_pred_12)

                    data_test_3 = test_3.to_dict(orient='records')
                    data_test_6 = test_6.to_dict(orient='records')
                    data_test_12 = test_12.to_dict(orient='records')

                    return render_template('train/index.html', data_test_3=data_test_3, data_test_6=data_test_6, data_test_12=data_test_12, mape_3=mape_3_persen, mape_6=mape_6_persen, mape_12=mape_12_persen, akurasi_3=akurasi_3, akurasi_6=akurasi_6, akurasi_12=akurasi_12)

                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html')

        except Exception as e:
            print(f"Error in index method: {e}")
            flash('An unexpected error occurred.')
            return redirect('/latih-model')

