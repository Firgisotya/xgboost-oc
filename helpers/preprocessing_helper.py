import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class PreprocessingHelper:
    def __init__(self):
        pass

    # load train data
    def load_data_train(self):
        file_path = os.path.join('static', 'dataset', 'dt_train.csv')
        return pd.read_csv(file_path)

    def load_data_test(self):
        file_path = os.path.join('static', 'dataset', 'dt_test.csv')
        return pd.read_csv(file_path)

    def dataset_train(self):
        file_path = os.path.join('static', 'dataset', 'data_hour_2023_reject.csv')
        return pd.read_csv(file_path)
    
    # drop kolom yang tidak diperlukan
    def load_dataset(self):
        train_dataset = self.load_data_train()
        test_dataset = self.load_data_test()

        train_dataset.drop(['lotno2', 'prod_order2'], axis=1, inplace=True)
        test_dataset.drop(['lotno2', 'prod_order2'], axis=1, inplace=True)

        train_dataset['tanggal'] = pd.to_datetime(train_dataset['tanggal'])
        test_dataset['tanggal'] = pd.to_datetime(test_dataset['tanggal'])

        X_train = train_dataset['value']
        y_train = train_dataset['reject']
        X_test = test_dataset['value']
        y_test = test_dataset['reject']
        return X_train, X_test, y_train, y_test
       
    
   
