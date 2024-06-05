import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import joblib
from helpers.model_helper import xgboost_model

class RandomizedSearch:
    def __init__(self, model, param_distributions, X_train, X_test, y_train, y_test, n_iter=10, cv=5, n_jobs=-1):
        self.model = model
        self.param_distributions = param_distributions
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs

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
    
    def save_model(self, solution, path):
        model = xgboost_model(solution, self.X_train)
        model.fit(self.X_train, self.y_train)
        joblib.dump(model, path)