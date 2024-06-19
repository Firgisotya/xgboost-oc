from flask import request, redirect, url_for, render_template

class HomeController:

    def index(self):
        
        return render_template('home/index.html')