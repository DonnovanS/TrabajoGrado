import requests
import json
from os import listdir
# initialize the Keras REST API endpoint URL along with the input
# image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = None
def ls(ruta='./Plantas/'): 
    return listdir(ruta)
contador = ls()
saved=len(contador)
print (saved)


for i in range(1,saved):
    IMAGE_PATH='./Plantas/imagen '+str(i)+'.JPG'
    
# load the input image and construct the payload for the request
    image = open(IMAGE_PATH, "rb").read()
    payload = {"image": image}

# submit the request
    r = requests.post(KERAS_REST_API_URL, files=payload).json()
    print(r)


