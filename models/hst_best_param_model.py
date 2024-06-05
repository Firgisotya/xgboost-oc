from config.db import connectdb

class HstBestParamModel:
    def __init__(self):
        self.connection = connectdb()

    def find_all(self):
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM hst_best_params")
                    rows = cursor.fetchall()
                    return rows
            else:
                print("Failed to connect to database.")
        except Exception as e:
            print(f"Error: {e}")

    def find_by_id(self, id):
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT * FROM hst_best_params WHERE id=%s", (id,))
                    row = cursor.fetchone()
                    return row
            else:
                print("Failed to connect to database.")
        except Exception as e:
            print(f"Error: {e}")

    def create(self, data):
        try:
            if self.connection:
                with self.connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO hst_best_params (learning_rate, max_depth, n_estimators, min_split_loss, mape) VALUES (%s, %s, %s, %s, %s)",
                        (
                            data["learning_rate"],
                            data["max_depth"],
                            data["n_estimators"],
                            data["min_split_loss"],
                            data["mape"],
                        ),
                    )
                    self.connection.commit()
            else:
                print("Failed to connect to database.")
        except Exception as e:
            print(f"Error: {e}")