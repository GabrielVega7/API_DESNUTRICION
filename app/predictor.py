# ==========================================================
# predictor.py
# ==========================================================

import joblib
import pandas as pd

from app.config import MODEL_PATH


modelo = joblib.load(MODEL_PATH)


print("Modelo cargado correctamente")


FEATURES = modelo.feature_names_in_.tolist()



class Predictor:


    modelo = modelo


    @staticmethod
    def features():

        return FEATURES



    @staticmethod
    def validar(datos):

        from app.utils import validar_variables

        return validar_variables(

            datos,

            FEATURES

        )



    @staticmethod
    def predict(datos):


        df = pd.DataFrame([datos])


        # mismo orden usado en entrenamiento
        df = df[FEATURES]


        pred = modelo.predict(df)[0]


        prob = modelo.predict_proba(df)[0].max()


        return int(pred), float(prob)