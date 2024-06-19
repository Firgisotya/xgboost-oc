from flask import request, redirect, url_for, render_template, flash
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from models.hst_best_param_model import HstBestParamModel
from helpers.preprocessing_helper import PreprocessingHelper
from helpers.model_helper import xgboost_model

class AnalisisController :
    
    def __init__(self):
        self.hstBestParamsModel = HstBestParamModel()
        self.preprocessing_helper = PreprocessingHelper()

    def index(self):
        try:
            hst_best_param = self.hstBestParamsModel.find_all()
            data = None
            

            if request.method == 'POST':
                try:

                    model = request.form['model']
                    data_uji = request.files['data_uji']

                    #simpan file sementara
                    file_path = os.path.join('static', 'uploads', data_uji.filename)
                    data_uji.save(file_path)

                    #load data uji
                    df_train = self.preprocessing_helper.load_data_train()
                    df_train['tanggal'] = pd.to_datetime(df_train['tanggal'])
                    df_train.set_index('tanggal', inplace=True)
                    df_train.dropna(inplace=True)
                    df_test = pd.read_csv(file_path)
                    df_test['tanggal'] = pd.to_datetime(df_test['tanggal'])
                    df_test.set_index('tanggal', inplace=True)
                    df_test.dropna(inplace=True)

                    
                    X_train = df_train.drop(columns=['lotno2', 'prod_order2', 'value'])
                    y_train = df_train['value']
                    X_test = df_test.drop(columns=['lotno2', 'prod_order2', 'value'])
                    # y_test = df_test['value']
                    
                    row = self.hstBestParamsModel.find_by_id(model)
                    params_distribution = [row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]]
                    # print(params_distribution)

                    model = xgboost_model(params_distribution)
                    model.fit(X_train, y_train)
                    prediksi = model.predict(X_test)
                    prediksi = np.round(prediksi)

                    df_test['value'] = prediksi

                    data = df_test
                    print(data)
                    

                    flash('Data berhasil disimpan', 'success')
                    return render_template('analisis/index.html', data=data.to_dict(orient='records'))
                except Exception as e:
                    print(f"Error: {e}")
                    flash('Data gagal disimpan', 'error')
                    return render_template('analisis/index.html')

        except Exception as e:
            print(f"Error: {e}")  
            return render_template('analisis/index.html')
   
    