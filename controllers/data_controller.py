from flask import request, redirect, url_for, render_template
from config.db import connectdb
from models.data_model import DataModel
from helpers.preprocessing_helper import PreprocessingHelper

class DataController :
    def __init__(self):
        self.data_model = DataModel()
        self.preprocessing_helper = PreprocessingHelper()
    
    def index(self):
        data = self.preprocessing_helper.dataset_train()
        return render_template("data/index.html", data=data)

        