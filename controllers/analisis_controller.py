from flask import request, redirect, url_for, render_template, flash
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
from helpers.preprocessing_helper import PreprocessingHelper

class AnalisisController:
    
    def __init__(self):
        self.preprocessing_helper = PreprocessingHelper()

    def load_dataset(self):
        file_path = os.path.join('static', 'dataset', 'data_hour_2023.csv')
        return pd.read_csv(file_path)

    def index(self):
        try:
            data_plot = {
                "label": [],
                "reject": [],
            }

            if request.method == 'POST':
                try:

                    # model = request.form['model']
                    data_uji = request.files['data_uji']

                    #simpan file sementara
                    file_path = os.path.join('static', 'uploads', data_uji.filename)
                    data_uji.save(file_path)

                    #load data uji
                    df_train = self.load_dataset()
                    df_train['tanggal'] = pd.to_datetime(df_train['tanggal'])
                    df_train.dropna(inplace=True)
                    df_test = pd.read_csv(file_path)
                    df_test['tanggal'] = pd.to_datetime(df_test['tanggal'])
                    df_test.dropna(inplace=True)

                    
                    X_train = df_train.drop(columns=["tanggal", "reject", "lotno2", "prod_order2"])
                    y_train = df_train["reject"]
                    X_test = df_test.drop(columns=["tanggal", "reject", "lotno2", "prod_order2"])
                    y_test = df_test["reject"]

                    # model
                    file_load = os.path.join('static', 'models', 'model_xgb.pkl')
                    model = joblib.load(file_load)
                    model.fit(X_train, y_train)

                    # prediksi
                    y_pred = model.predict(X_test)
                    y_pred = np.round(y_pred)

                    # MAPE
                    # mape = mean_absolute_percentage_error(y_test, y_pred)

                    # mape_persen = mape * 100

                    # akurasi = 100 - mape_persen

                    # print(f"MAPE: {round(mape, 3)}")

                    # print(f"MAPE (%): {mape_persen:.2f}%")

                    # print(f"Akurasi (%): {akurasi:.2f}%")

                    df_test["reject"] = np.round(y_pred)

                    data_test = df_test.to_dict(orient='records')

                    data_plot["label"] = df_test["lotno2"].tolist()
                    data_plot["reject"] = df_test["reject"].tolist()
                    
                    return render_template('analisis/index.html', data=data_test, data_plot=data_plot)
                except Exception as e:
                    print(f"Error: {e}")
                    flash('Data gagal disimpan', 'error')
                    return render_template('analisis/index.html')
            
            return render_template('analisis/index.html')

        except Exception as e:
            print(f"Error: {e}")  
            return render_template('analisis/index.html')
   
    