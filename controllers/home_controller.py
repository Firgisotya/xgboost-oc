from flask import request, redirect, url_for
from helper import render, view
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from config.db import connectdb

class HomeController:

    def index(self):
        
        return render(view('home/index'))