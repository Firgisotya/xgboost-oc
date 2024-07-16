from flask import render_template, request, redirect, session, flash
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from helpers.preprocessing_helper import PreprocessingHelper


class TrainModelController:
    def __init__(self):
        self.preprocessing_helper = PreprocessingHelper()

    def load_data_test(self):
        file_path = os.path.join('static', 'dataset', 'df_test.csv')
        return pd.read_csv(file_path)
    
    def index(self):
        try:
            mape = None
            data_plot = {
                "label": [],
                "reject": [],
                "forecast": []
            }

            if request.method == 'POST':
                try:
                    X_train, X_test, y_train, y_test = self.preprocessing_helper.load_dataset()

                    # model
                    model = xgb.XGBRegressor(
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

                    model.fit(X_train, y_train)

                    # save model
                    model_path = os.path.join('static', 'models', 'model_xgb.pkl')
                    joblib.dump(model, model_path)
                    
                    # prediksi
                    y_pred = model.predict(X_test)
                    y_pred = np.round(y_pred)

                    # MAPE
                    mape = mean_absolute_percentage_error(y_test, y_pred)

                    mape_persen = mape * 100

                    akurasi = 100 - mape_persen

                    print(f"MAPE: {round(mape, 3)}")

                    print(f"MAPE (%): {mape_persen:.2f}%")

                    print(f"Akurasi (%): {akurasi:.2f}%")

                    test = self.load_data_test()

                    test["forecast"] = np.round(y_pred)

                    data_test = test.to_dict(orient='records')

                    data_plot["label"] = test["lotno2"].tolist()
                    data_plot["reject"] = test["reject"].tolist()
                    data_plot["forecast"] = test["forecast"].tolist()

                    print(data_plot)

                    return render_template('train/index.html', data=data_test, data_plot=data_plot, mape=mape_persen, akurasi=akurasi)

                except Exception as e:
                    print(e)
                    flash('There was an error processing the form. Please check your input values.')
                    return redirect('/latih-model')

            return render_template('train/index.html')

        except Exception as e:
            print(f"Error in index method: {e}")
            flash('An unexpected error occurred.')
            return redirect('/latih-model')

