from flask import Blueprint
from controllers.home_controller import HomeController
from controllers.data_controller import DataController
from controllers.analisis_controller import AnalisisController
from controllers.train_model_controller import TrainModelController

homeController = HomeController()
dataController = DataController()
analisisController = AnalisisController()
trainModelController = TrainModelController()

web = Blueprint('web', __name__)

web.add_url_rule('/', view_func=homeController.index, methods=['GET'], endpoint='home')
web.add_url_rule('/data', view_func=dataController.index, methods=['GET'], endpoint='data')
@web.route('/analisis', methods=['GET', 'POST'])
def analisis():
    return analisisController.index()

@web.route('/latih-model', methods=['GET', 'POST'])
def train():
    return trainModelController.index()