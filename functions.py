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


def deteccion(image):
    
    

    

   
    
    
    model = YOLO("bestNDVI.pt")
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




