from pathlib import Path

# Carpeta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta donde se encuentra el modelo entrenado
MODEL_PATH = BASE_DIR / "models" / "modelo_final_api.joblib"
DATA_PATH = "data/ENSANUT_MODELO.csv"

# Información de la API
API_TITLE = "API de Predicción de Desnutrición Crónica Infantil"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
API REST desarrollada con FastAPI para predecir
el riesgo de desnutrición crónica infantil mediante
un modelo de Machine Learning (XGBoost).
"""