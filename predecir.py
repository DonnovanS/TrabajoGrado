import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

longitud, altura=100,100                    #misma longitud definida en entrenamiento
modelo= './modelo/modelo.h5'                #ruta de modelo
pesos= './modelo/pesos.h5'                  #ruta de pesos
cnn=load_model(modelo)                      #cnn carga la variable modelo
cnn.load_weights(pesos)                     #carga de pesos

def predict(file):              #recibir nombre de imagen y definir que es 
    X=load_img(file, target_size=(longitud, altura))       
    X=img_to_array(X)                                       #convertir en arreglo la imagen
    X=np.expand_dims(X, axis=0)                             #procesamiento de información
    arreglo=cnn.predict(X) ##[[1,0,0]]                      #predicción de imagen en 2D 
    resultado=arreglo[0] ##[1,0,0]                          #predice una dimensión     
    respuesta=np.argmax(resultado) #[0]                     #indice de valor mas alto que se tenga en resultado
    if respuesta==0:                                
        print('Planta Sana')
    elif respuesta==1:
        print('Mildiu')
    elif respuesta==2:
        print('Mildiu')
    return respuesta

predict('planta.jpeg')
    