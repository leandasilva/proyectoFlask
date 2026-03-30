from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/imagenes/.gitkeep'

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

# Cargar el modelo entrenado
modelo = load_model('modelo.h5')

# Tamaño de entrada del modelo
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

    # Validar extensión de imagen
    if not archivo.filename.lower().endswith(('.jpg', '.jpeg')):
        return "Formato de imagen no permitido. Solo se aceptan .jpg o .jpeg", 400

    ruta_guardada = os.path.join(app.config['UPLOAD_FOLDER'], archivo.filename)
    archivo.save(ruta_guardada)
    
    # Preprocesar la imagen
    img = image.load_img(ruta_guardada, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Realizar la predicción
    prediccion = modelo.predict(img_array)
    indice_predicho = np.argmax(prediccion[0])
    resultado = clases.get(indice_predicho, "Desconocido")

    print(f"Predicción numérica: {indice_predicho}, clase: {resultado}")

    return render_template('resultado.html', imagen=ruta_guardada, resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
