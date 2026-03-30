from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/imagenes'

# Diccionario de clases
clases = {
    0: 'avion',
    1: 'automovil',
    2: 'pajaro',
    3: 'gato',
    4: 'ciervo',
    5: 'perro',
    6: 'rana',
    7: 'caballo',
    8: 'barco',
    9: 'camion'
}

# 🔥 Ruta base segura
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 Cargar modelo de forma segura
modelo = None

def get_model():
    global modelo
    if modelo is None:
        ruta_modelo = os.path.join(BASE_DIR, "modelo.h5")
        modelo = load_model(ruta_modelo)
    return modelo

# Tamaño de entrada
IMG_SIZE = (32, 32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    if 'imagen' not in request.files:
        return "No se subió ningún archivo", 400

    archivo = request.files['imagen']
    if archivo.filename == '':
        return "Nombre de archivo vacío", 400

    if not archivo.filename.lower().endswith(('.jpg', '.jpeg')):
        return "Formato no permitido (.jpg/.jpeg)", 400

    # 🔥 Crear carpeta si no existe
    carpeta = os.path.join(BASE_DIR, app.config['UPLOAD_FOLDER'])
    os.makedirs(carpeta, exist_ok=True)

    ruta_guardada = os.path.join(carpeta, archivo.filename)
    archivo.save(ruta_guardada)

    # Preprocesar imagen
    img = image.load_img(ruta_guardada, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # 🔥 Obtener modelo (lazy load)
    model = get_model()

    # Predicción
    prediccion = model.predict(img_array)
    indice_predicho = np.argmax(prediccion[0])
    resultado = clases.get(indice_predicho, "Desconocido")

    print(f"Predicción: {resultado}")

    # Ruta relativa para HTML
    ruta_relativa = f"static/imagenes/{archivo.filename}"

    return render_template('resultado.html', imagen=ruta_relativa, resultado=resultado)


# ❌ NO necesario en Render, pero lo dejamos para local
if __name__ == '__main__':
    app.run(debug=True)