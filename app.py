from flask import Flask
from routes.web_route import web
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
# import models

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

def check_secret_key():
    secret_key = os.getenv('SECRET_KEY')
    if secret_key:
        return True
    else:
        return False


# Register for WEB
app.register_blueprint(web)

if __name__ == '__main__':
    # models.Model().test_connection()
    if check_secret_key() == False:
        print('you have not generated a key, please generate a key with `python3 config/generate_key.py`')
    else:
        app.run(
            host='0.0.0.0', 
            port=os.getenv('APP_PORT'),
            debug=True if os.getenv('APP_DEBUG') == 'development' else False
        )
