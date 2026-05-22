import tensorflow as tf
import numpy as np
import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# --- FIX softmax_v2 ---
@tf.keras.utils.register_keras_serializable()
def softmax_v2(x):
    return tf.nn.softmax(x)

app = FastAPI(title="API Detección de Anemia")


try:
    modelo_nn = tf.keras.models.load_model(
        "modelo_anemia.keras",
        custom_objects={"softmax_v2": softmax_v2}
    )
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")


class DatosPaciente(BaseModel):
    sw: int
    Sexo: int
    EdadMeses: int
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


@app.post("/predict")
def predecir(datos: DatosPaciente):
    entrada = np.array([[
        datos.sw, datos.Sexo, datos.EdadMeses, datos.Juntos, datos.SIS,
        datos.Qaliwarma, datos.Hemoglobina, datos.Cred, datos.Suplementacion,
        datos.Consejeria, datos.Sesion, datos.AlturaREN, datos.Hbc
    ]])

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



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)