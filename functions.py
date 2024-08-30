import streamlit as st
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np
import imutils
import os
from os import mkdir
from datetime import date
from datetime import datetime
from getpass import getuser
import supervision as sv
from PIL import Image

import matplotlib.path as mplPath
import matplotlib.pyplot as plt



from PIL import Image

from ultralytics import YOLO
from ultralytics.solutions import object_counter

import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import imutils


# Define some helper functions for downloading pretrained model
# taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
import requests
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)  

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def load_model(self):
        cloud_model_location = "1jbDLmw_ZWjDgUlVGrVde4h68U1edPzdT"
        save_dest = Path('model')
        save_dest.mkdir(exist_ok=True)

        f_checkpoint = Path("bestNDVI.pt")

        if not f_checkpoint.exists():
            with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
                from GD_download import download_file_from_google_drive
                download_file_from_google_drive(cloud_model_location, f_checkpoint)

        checkpoint = torch.load(f_checkpoint, map_location=device)
        self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
        self.net_G.to(device)
        self.net_G.eval()
        
        # To free memory!
        del f_checkpoint
        del checkpoint
        
        #return model    




def deteccion(image):
    
    

    

   
    
    
    model = YOLO("Models/bestNDVI.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.1, show_labels=False,show_conf=False)[0]
    resultados = model.predict(imagen, imgsz = 640, conf = 0.1)
    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.1]
    
    leng = len(resultados)
    
    anotaciones = resultados[0].plot()
    haber = imutils.resize(anotaciones,width=640)
    #cv2.imshow("Resutados de la deteccion", haber)
    
    haber = imutils.resize(anotaciones,width=1024)
    #cv2.imshow("Resultados " + fecha, haber)
    #cv2.imwrite(filename, haber)
    #lblInfo2 = Label(root,text="Anomalías detectadas: " + leng2)
    #lblInfo2.grid(column=0,row=2,padx=5,pady=5)
    
    
    
    return haber



def deteccion2(image):
    
    model = YOLO("Models/bestNDVI.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.1, show_labels=False,show_conf=False)[0]
    resultados = model.predict(imagen, imgsz = 640, conf = 0.1)
    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.1]
    
    leng = len(resultados)
    
    anotaciones = resultados[0].plot()
    haber = imutils.resize(anotaciones,width=640)
    #cv2.imshow("Resutados de la deteccion", haber)
    leng = len(alta)
    leng2 = str(leng)
    
    haber = imutils.resize(anotaciones,width=1024)
    #cv2.imshow("Resultados " + fecha, haber)
    #cv2.imwrite(filename, haber)
    #lblInfo2 = Label(root,text="Anomalías detectadas: " + leng2)
    #lblInfo2.grid(column=0,row=2,padx=5,pady=5)
    
    
    return leng2




