import os
from dotenv import load_dotenv
import pymysql

def connectdb():
        try:
            load_dotenv()
            connection = pymysql.connect(
                host=os.getenv('DB_HOST'),
                user=os.getenv('DB_USER'),
                password=os.getenv('DB_PASS'),
                db=os.getenv('DB_NAME'),
                port=int(os.getenv('DB_PORT')),
            )
            return connection
        except pymysql.MySQLError as e:
            print(f"Error connecting to database: {e}")
            return None
