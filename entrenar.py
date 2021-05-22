import sys
import os
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D
from tensorflow.python.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import argparse


tf.keras.callbacks.History()
tf.compat.v1.disable_eager_execution()
K.clear_session()

data_entrenamiento='./data/entrenamiento'
data_validacion='./data/validación'

#Parametros


epocas=15
altura, longitud= 100, 100
batch_size=32
pasos=1000
pasos_validacion=200
filtrosConv1=32
filtrosConv2=64
tamano_filtro1=(3,3)
tamano_filtro2=(2,2)
tamano_pool=(2,2)
clases=2
lr=0.0005

#pre procesamiento de imagenes

entrenamiento_datagen= ImageDataGenerator(
    rescale=1./255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

validacion_datagen=ImageDataGenerator(
    rescale=1./255
)

imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Red Neuronal 

cnn=Sequential()

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud,3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())
cnn.add(Dense(256,activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(clases, activation='softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])

result=cnn.fit(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)


dir='./modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')
cnn.save_weights('./modelo/pesos.h5')

plt.figure()
plt.plot(np.arange(0, 15), result.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), result.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), result.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 15), result.history["val_accuracy"], label="val_acc")
plt.title('perdidas del modelo')
plt.ylabel('perdida')
plt.xlabel('epoca')
plt.legend()
plt.show()
predictions=cnn.predict_generator(imagen_validacion)

y_pred = np.argmax(predictions, axis=1)
class_labels =list(imagen_validacion.class_indices.keys())
print('\n​Confusion Matrix​\n')
print(class_labels)
print(confusion_matrix(imagen_validacion.classes, y_pred))
# Confusion Metrics: Accuracy, Precision, Recall & F1 Score
report=classification_report(imagen_validacion.classes,y_pred, target_names=["Enfermas", "Sanas"])
print('\n​Classification Report​\n')
print(report)

plot_model(cnn, to_file="AlgoritmoD.png", show_shapes=True)