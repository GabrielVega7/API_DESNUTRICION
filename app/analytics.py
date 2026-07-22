# ==========================================================
# analytics.py
# ==========================================================

import pandas as pd
import numpy as np

from app.config import DATA_PATH
from app.predictor import Predictor


class Analytics:


    # ======================================================
    # MÉTRICAS DEL MODELO
    # ======================================================

    @staticmethod
    def metricas():

        return {

            "modelo":"XGBoost",

            "accuracy":0.767918,

            "precision":0.713174,

            "recall":0.767918,

            "f1_score":0.718441

        }



    # ======================================================
    # IMPORTANCIA DE VARIABLES
    # ======================================================

    @staticmethod
    def importancia():

        pipeline = Predictor.modelo

        modelo = pipeline.named_steps["modelo"]

        preprocesador = pipeline.named_steps["preprocesador"]


        # Obtener importancia del modelo

        importancia = modelo.feature_importances_


        # Obtener variables transformadas

        variables = preprocesador.get_feature_names_out()


        print(
            "Variables transformadas:",
            len(variables)
        )

        print(
            "Importancias:",
            len(importancia)
        )


        # Ajustar si existe diferencia

        minimo = min(
            len(variables),
            len(importancia)
        )


        resultado = pd.DataFrame({

            "variable": variables[:minimo],

            "importancia": importancia[:minimo]

        })


        resultado = resultado.sort_values(
            by="importancia",
            ascending=False
        )


        return resultado.head(15).to_dict(
            orient="records"
        )



    # ======================================================
    # NIVEL DE RIESGO
    # ======================================================

    @staticmethod
    def riesgo():

        df = pd.read_csv(DATA_PATH)


        X = df[Predictor.features()]


        probabilidades = Predictor.modelo.predict_proba(X)[:,1]


        alto = int(
            (probabilidades >= 0.80).sum()
        )


        medio = int(
            ((probabilidades >=0.50) &
             (probabilidades <0.80)).sum()
        )


        bajo = int(
            (probabilidades <0.50).sum()
        )


        return {


            "riesgo_alto":alto,

            "riesgo_medio":medio,

            "riesgo_bajo":bajo

        }