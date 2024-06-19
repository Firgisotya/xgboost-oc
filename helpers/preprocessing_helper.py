import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocessingHelper:
    def __init__(self):
        pass

    # load train data
    def load_data_train(self):
        file_path = os.path.join('static', 'dataset', 'data_hour_2023.csv')
        return pd.read_csv(file_path)
    
    # drop kolom yang tidak diperlukan
    def load_dataset(self):
        data = self.load_data_train()
        data['tanggal'] = pd.to_datetime(data['tanggal'])
        data.set_index('tanggal', inplace=True)
        data.dropna(inplace=True)
        data.drop(['prod_order2', 'lotno2'], axis=1, inplace=True)
        X = data.drop('value', axis=1)
        y = data['value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
       
    
   
