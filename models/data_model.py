from flask import request, redirect, url_for, render_template
from config.db import connectdb

class DataModel:
    def __init__(self):
        self.connection = connectdb()
    
    def find_all(self):
        data = []
        try:
            connection = connectdb()
            if connection:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM tmp_data")
                    rows = cursor.fetchall()
                    for row in rows:
                        formatted_row = {
                            "tanggal": row[0],
                            "lotno2": row[1],
                            "prod_order2": row[2],
                            "value": row[3],
                            "temp_mini_tank": row[4],
                            "temp_nozzle": row[5],
                            "temp_nozzle_filler": row[6],
                            "label": row[7],
                        }
                        data.append(formatted_row)
                connection.close()
            else:
                print("Failed to connect to database.")
        except Exception as e:
            print(f"Error: {e}")
        return data
