import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class AlgoritmaHelper:
    def __init__(self):
        pass

    # load train data
    def load_data_train(self):
        file_path = os.path.join('dataset', 'data_hour_2023')
        return pd.read_csv(file_path)
    
    # drop kolom yang tidak diperlukan
    def load_dataset (self):
        data = self.load_data_train()
        data.dropna(inplace=True)
        data.drop(['prod_order2', 'lotno2'], axis=1, inplace=True)
        X = data.drop('value', axis=1)
        y = data['value']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    # get best params with randomized search
    def param_distributions(learning_rate, max_depth, n_estimators, min_split_loss):
        learning_rate = float(learning_rate)
        max_depth = max(20, min(7, int(max_depth)))
        n_estimators = max(600, min(300, int(n_estimators)))
        min_split_loss = max(3, min(1, int(min_split_loss)))
        return {
            'learning_rate': [learning_rate],
            'max_depth': [max_depth],
            'n_estimators': [n_estimators],
            'min_split_loss': [min_split_loss]
        }
        # return param_distributions

    # get best model with randomized search
    def rscv_generator(self, model, param_distributions, X_train, X_test, y_train, y_test):
        rscv = RandomizedSearch(model, param_distributions, X_train, X_test, y_train, y_test)
        return rscv
    
    # get best model with randomized search
    def optimal_params(self):
        rs = RandomizedSearchCV(
            self.model,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            n_jobs=self.n_jobs
        )

        rs.fit(self.X_train, self.y_train)

        return rs.best_params_
    
    
    

