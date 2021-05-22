import flask
import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow import keras
from keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import io
import json

longitud, altura = 100, 100
modelo = '.Proyecto/modelo/modelo.h5'
pesos = '.Proyecto/modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)

app = flask.Flask(__name__)


def predict(file):
   # X=load_img(file, target_size=(longitud, altura))
    X = file.resize((100, 100))
    X = img_to_array(X)
    X = np.expand_dims(X, axis=0)
    arreglo = cnn.predict(X)  # [[1,0,0]]
    resultado = arreglo[0]  # [1,0,0]
    respuesta = np.argmax(resultado)  # [0]
    if respuesta == 0:
        print('La planta padece de Mildiu Velloso')
    elif respuesta == 1:
        print('La planta es sana')
    return respuesta

@app.route("/predict", methods=["POST"])
def predic():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))
			image = predict(image)
			# data["predictions"] = []
			prediccion= 'Planta Enferma' if (image==0) else 'Planta Sana'
			print(prediccion)
			r = {'respuesta': prediccion} 
			# data["predictions"].append(r)
			# data["success"] = True
			#records = json.loads(r)
            # indicate that the request was a success
			# data["success"] = True
	# return the data dictionary as a JSON response
	return flask.jsonify(r)

if __name__=='__main__':
    app.run()
