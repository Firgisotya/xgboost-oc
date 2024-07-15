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

        # Extract the month and year from the 'tanggal' column
        dataset['month'] = dataset['tanggal'].dt.month
        dataset['year'] = dataset['tanggal'].dt.year

        # Split the dataset into 3, 6, 9, and 12 months
        df_3 = dataset[(dataset['year'] == 2023) & (dataset['month'] <= 3)]
        df_6 = dataset[(dataset['year'] == 2023) & (dataset['month'] <= 6)]
        df_12 = dataset[dataset['year'] == 2023]

        # menghapus data yang bernilai null
        df_3.dropna(inplace=True)
        df_6.dropna(inplace=True)
        df_12.dropna(inplace=True)

        train_3 = df_3[: int(0.8 * (len(df_3)))]
        test_3 = df_3[int(0.8 * (len(df_3))) :]

        train_6 = df_6[: int(0.8 * (len(df_6)))]
        test_6 = df_6[int(0.8 * (len(df_6))) :]

        train_12 = df_12[: int(0.8 * (len(df_12)))]
        test_12 = df_12[int(0.8 * (len(df_12))) :]

        # simpan data train dan data test ke folder static dataset dalam bentuk csv
        train_3.to_csv(os.path.join('static', 'dataset', 'df_train_3.csv'), index=False)
        test_3.to_csv(os.path.join('static', 'dataset', 'df_test_3.csv'), index=False)

        train_6.to_csv(os.path.join('static', 'dataset', 'df_train_6.csv'), index=False)
        test_6.to_csv(os.path.join('static', 'dataset', 'df_test_6.csv'), index=False)

        train_12.to_csv(os.path.join('static', 'dataset', 'df_train_12.csv'), index=False)
        test_12.to_csv(os.path.join('static', 'dataset', 'df_test_12.csv'), index=False)


        # inisiasi X_train, X_test, y_train, y_test
        X_train_3 = train_3.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_train_3 = train_3["reject"]
        X_test_3 = test_3.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_test_3 = test_3["reject"]

        X_train_6 = train_6.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_train_6 = train_6["reject"]
        X_test_6 = test_6.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_test_6 = test_6["reject"]

        X_train_12 = train_12.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_train_12 = train_12["reject"]
        X_test_12 = test_12.drop(columns=["Unnamed: 0", "tanggal", "reject", "lotno2", "prod_order2", "month", "year"])
        y_test_12 = test_12["reject"]

        return X_train_3, X_test_3, y_train_3, y_test_3, X_train_6, X_test_6, y_train_6, y_test_6, X_train_12, X_test_12, y_train_12, y_test_12