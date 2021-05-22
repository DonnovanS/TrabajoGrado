import keras
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from tensorflow import keras
from keras.initializers import glorot_uniform
from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
from imutils import paths

longitud, altura = 100, 100
modelo = './modelo/modelo.h5'
pesos = './modelo/pesos.h5'
cnn = load_model(modelo)
cnn.load_weights(pesos)
classLabels = ["Planta Enferma", "Planta Sana"]


def predict(file,index):
	X = load_img(file, target_size=(longitud, altura))
	X = img_to_array(X)
	X = np.expand_dims(X, axis=0)
	arreglo = cnn.predict(X)  # [[1,0,0]]
	resultado = arreglo[0]  # [1,0,0]
	respuesta = np.argmax(resultado)  # [0]
	imagePaths = list(paths.list_images('./Plantas'))
#	try:
		#for (i, imagePath) in enumerate(imagePaths):
 		# load the example image, draw the prediction, and display it
		# to our screen
#			image = cv2.imread(file)
#			print(image[index])
#			cv2.putText(image, "Label: {}".format(classLabels[arreglo.argmax(axis=1)[0]]),
 #   	    	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
	#		cv2.imshow("Image", image)
	#		cv2.waitKey(0)
	#except:
	#	print('Error')
	if respuesta == 0:
		print('La planta padece de Mildiu Velloso')
	elif respuesta == 1:
		print('La planta es sana')
	return respuesta

for i in range(1,29):
	predict('./Plantas/imagen '+str(i)+'.JPG',i)
