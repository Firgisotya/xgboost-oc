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
        file_path = os.path.join('static', 'dataset', 'data_hour_2023.csv')
        return pd.read_csv(file_path)
        
    
    # clenaning and selection data
    def load_dataset(self):
        # load data
        dataset = self.dataset_train()

        # Convert the 'tanggal' column to datetime format
        dataset['tanggal'] = pd.to_datetime(dataset['tanggal'])

        # menghapus data yang bernilai null
        dataset.dropna(inplace=True)

        train = dataset[: int(0.8 * (len(dataset)))]
        test = dataset[int(0.8 * (len(dataset))) :]
        # simpan data train dan data test ke folder static dataset dalam bentuk csv
        train.to_csv(os.path.join('static', 'dataset', 'df_train.csv'), index=False)
        test.to_csv(os.path.join('static', 'dataset', 'df_test.csv'), index=False)


        # inisiasi X_train, X_test, y_train, y_test
        X_train = train.drop(columns=["tanggal", "reject", "lotno2", "prod_order2"])
        y_train = train["reject"]
        X_test = test.drop(columns=["tanggal", "reject", "lotno2", "prod_order2"])
        y_test = test["reject"]

        return X_train, X_test, y_train, y_test