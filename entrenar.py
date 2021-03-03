import sys                  #Librerias para navegar entre carpetas del equipo
import os
import tensorflow as tf
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator # preprocesamiento de imagen
from tensorflow.python.keras import optimizers     #Optimizador para entrenar algoritmo  
from tensorflow.python.keras.models import Sequential    #Realizar redes neuronales secuenciales en orden
from tensorflow.python.keras.layers import Dropout, Flatten, Dense   #
from tensorflow.python.keras.layers import  Convolution2D, MaxPooling2D  #aplicación de filtros 
from tensorflow.python.keras import backend as K  #elimina procesos de sesiones de keras corriendo 

tf.compat.v1.disable_eager_execution()
K.clear_session()              

data_entrenamiento='./data/entrenamiento'        #variable de entrenamiento directorio
data_validacion='./data/validación'              #variable de validación 

##Parametros

epocas=10                          #numero de veces de iteraciones 
altura, longitud= 100, 100         #tamaño a procesar las imagenes 
batch_size=32                      #numero de imagenes a procesar en cada paso 
pasos=500                          #numero de veces que se procesa info en cada epoca
pasos_validacion=200               #ejecuta numero de pasos para validación
filtrosConv1=32                    #numero de filtros aplicados en cada convolución profundidad de 32
filtrosConv2=64                    #numero de filtros aplicados en cada convolución profundidad de 64
tamano_filtro1=(3,3)               #tamaño del filtro, altura y longitud
tamano_filtro2=(2,2)               #tamaño del filtro, altura y longitud
tamano_pool=(2,2)                  #tamaño de filtro en maxpooling
clases=3                           #numero de carpetas
lr=0.0005                          #tamaño de ajuste de red neuronal para acercarse a una predicción optima


##pre procesamiento de imagenes

entrenamiento_datagen= ImageDataGenerator(          
    rescale=1./255,                                 #re escalado de pixeles de 0 a 1
    shear_range=0.3,                                #generar las imagenes en inclinación determinada
    zoom_range=0.3,                                 #realiza zoom a imagen
    horizontal_flip=True                            #inversión de imagen para direccionalidad
)

validacion_datagen=ImageDataGenerator(  
    rescale=1./255  
)

imagen_entrenamiento= entrenamiento_datagen.flow_from_directory(            #ingresar a la data
    data_entrenamiento,                 #abrir todas las carpetas 
    target_size=(altura, longitud),      #procesa imagen
    batch_size=batch_size,
    class_mode='categorical'            
)

imagen_validacion=validacion_datagen.flow_from_directory(
    data_validacion,
    target_size=(altura, longitud),
    batch_size=batch_size,
    class_mode='categorical'
)

#Crear red Convolucional 

cnn=Sequential()            #varias capas 

cnn.add(Convolution2D(filtrosConv1, tamano_filtro1, padding='same', input_shape=(altura, longitud,3), activation='relu'))       #aplicación de primera capa y 3 canales de RGB

cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Convolution2D(filtrosConv2, tamano_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamano_pool))

cnn.add(Flatten())              #volver imagen plana
cnn.add(Dense(256,activation='relu'))       #256 neuronas despues de aplanar enviar a una capa normal
cnn.add(Dropout(0.5))                       #evitar sobre ajuste de manera aleatoria
cnn.add(Dense(clases, activation='softmax'))        #ultima capa de 3 neuronas para definir probabilidad de acierto 

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])    #parametros para optimizar el algoritmo, porcentaje aprendizaje

cnn.fit(imagen_entrenamiento,steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)     #entenamiento por epocas

dir='./modelo/'     #Guardar el modelo en un archivo

if not os.path.exists(dir):
    os.mkdir(dir)
cnn.save('./modelo/modelo.h5')          #Grabar modelo Estructura
cnn.save_weights('./modelo/pesos.h5')   #Grabar pesos de imagen 

