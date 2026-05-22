import tensorflow as tf
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# =========================
# FIX softmax custom layer
# =========================
@tf.keras.utils.register_keras_serializable()
def softmax_v2(x):
    return tf.nn.softmax(x)

# =========================
# FastAPI app
# =========================
app = FastAPI(title="API Detección de Anemia")

# =========================
# Load model
# =========================
modelo_nn = None

try:
    modelo_nn = tf.keras.models.load_model(
        "modelo_anemia.h5",
        custom_objects={"softmax_v2": softmax_v2}
    )
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    modelo_nn = None


# =========================
# Input schema
# =========================
class DatosPaciente(BaseModel):
    sw: int
    Sexo: int
    EdadMeses: float
    Juntos: int
    SIS: int
    Qaliwarma: int
    Hemoglobina: float
    Cred: int
    Suplementacion: int
    Consejeria: int
    Sesion: int
    AlturaREN: float
    Hbc: float


# =========================
# Predict endpoint
# =========================
@app.post("/predict")
def predecir(datos: DatosPaciente):

    # check model
    if modelo_nn is None:
        return {"error": "Modelo no cargado en el servidor"}

    # input vector (IMPORTANTE float32)
    entrada = np.array([[
        datos.sw,
        datos.Sexo,
        datos.EdadMeses,
        datos.Juntos,
        datos.SIS,
        datos.Qaliwarma,
        datos.Hemoglobina,
        datos.Cred,
        datos.Suplementacion,
        datos.Consejeria,
        datos.Sesion,
        datos.AlturaREN,
        datos.Hbc
    ]], dtype=np.float32)

    # prediction
    prediccion = modelo_nn.predict(entrada)
    id_clase = int(np.argmax(prediccion[0]))

    diagnosticos = {
        0: "Normal",
        1: "Anemia Moderada",
        2: "Anemia Leve",
        3: "Anemia Severa"
    }

    return {
        "resultado": diagnosticos.get(id_clase, "No identificado"),
        "probabilidades": prediccion[0].tolist()
    }