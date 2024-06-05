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


class AnalisisController :
    
    def __init__(self):
        self.hstBestParamsModel = HstBestParamModel()

    def index(self):
        try:
            hst_best_param = self.hstBestParamsModel.find_all()
            print(hst_best_param)

            return render_template('analisis/index.html', hst_best_param=hst_best_param)
        except Exception as e:
            print(f"Error: {e}")  
   
    