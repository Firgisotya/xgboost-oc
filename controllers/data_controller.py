from flask import request, redirect, url_for, render_template
from config.db import connectdb
from models.data_model import DataModel

class DataController :
    def __init__(self):
        self.data_model = DataModel()
    
    def index(self):
        data = self.data_model.find_all()
        return render_template("data/index.html", data=data)

        