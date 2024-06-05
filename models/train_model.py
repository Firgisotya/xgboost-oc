import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from config.db import connectdb
import joblib

class TrainModel :

    def fetch_dataset():
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
                        }
                        data.append(formatted_row)
                connection.close()
            else:
                print("Failed to connect to database.")
        except Exception as e:
            print(f"Error: {e}")
        return data


    def train_model(self):
        data = self.fetch_dataset()
        df = pd.DataFrame(data)
        df["tanggal"] = pd.to_datetime(df["tanggal"])
        df.set_index("tanggal", inplace=True)
        df.dropna(inplace=True)
        df.drop(["prod_order2", "lotno2"], axis=1, inplace=True)

        train = df[: int(0.8 * len(df))]
        test = df[int(0.8 * len(df)) :]

        X = train.drop("value", axis=1)
        y = train["value"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=123
        )

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            colsample_bytree=0.3,
            learning_rate=0.1,
            max_depth=5,
            alpha=10,
            n_estimators=10,
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        print("RMSE:", rmse)
        print("MAE:", mae)
