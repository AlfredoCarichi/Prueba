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
from PIL import ImageTk
import matplotlib.path as mplPath
import matplotlib.pyplot as plt



from PIL import Image

from ultralytics import YOLO
from ultralytics.solutions import object_counter

import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import imutils




def histograma(image):
    canales = cv2.split(image)
    colores = ('b','g','r')
    plt.figure()
    plt.title('Histograma de colores')
    plt.xlabel('Bits')
    plt.ylabel('# Píxeles')
    for (canal,color) in zip(canales,colores):
        hist = cv2.calcHist([canal],[0],None,[256],[0,256])
        plt.plot(hist,color=color)
        plt.xlim([0,256])
    plt.show()



def deteccion(image):
    user = getuser()
    print(user)

    f = open("Registros.txt","a")

    now = datetime.now()
    now2 = datetime.today()
    fecha = str(now)
    fecha2 = str(now2)

    route = os.getcwd()
    route2 = str(route)

    directory = route2
    os.chdir(directory)
    filename = 'Imagen_Resultados.jpg'
    
    
    model = YOLO("bestNDVI.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.1, show_labels=False,show_conf=False)[0]
    resultados = model.predict(imagen, imgsz = 640, conf = 0.1)
    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.1]
    print(resultados)
    leng = len(resultados)
    print(leng)
    anotaciones = resultados[0].plot()
    haber = imutils.resize(anotaciones,width=640)
    #cv2.imshow("Resutados de la deteccion", haber)
    leng = len(alta)
    leng2 = str(leng)
    print("------------------------------------")
    print("------------------------------------")
    print(leng2 + " Anomalías detectadas")
    haber = imutils.resize(anotaciones,width=1024)
    #cv2.imshow("Resultados " + fecha, haber)
    #cv2.imwrite(filename, haber)
    #lblInfo2 = Label(root,text="Anomalías detectadas: " + leng2)
    #lblInfo2.grid(column=0,row=2,padx=5,pady=5)
    f.write("\n--------Fecha del analisis : " + fecha2 + " Anomalías: " + leng2)
    f.close()
    img_conv = Image.fromarray(haber)
    
    return haber



def deteccion2(image):
    user = getuser()
    print(user)

    f = open("Registros.txt","a")

    now = datetime.now()
    now2 = datetime.today()
    fecha = str(now)
    fecha2 = str(now2)

    route = os.getcwd()
    route2 = str(route)

    directory = route2
    os.chdir(directory)
    filename = 'Imagen_Resultados.jpg'
    
    
    model = YOLO("Models/bestNDVI.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.1, show_labels=False,show_conf=False)[0]
    resultados = model.predict(imagen, imgsz = 640, conf = 0.1)
    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.1]
    print(resultados)
    leng = len(resultados)
    print(leng)
    anotaciones = resultados[0].plot()
    haber = imutils.resize(anotaciones,width=640)
    #cv2.imshow("Resutados de la deteccion", haber)
    leng = len(alta)
    leng2 = str(leng)
    print("------------------------------------")
    print("------------------------------------")
    print(leng2 + " Anomalías detectadas")
    haber = imutils.resize(anotaciones,width=1024)
    #cv2.imshow("Resultados " + fecha, haber)
    #cv2.imwrite(filename, haber)
    #lblInfo2 = Label(root,text="Anomalías detectadas: " + leng2)
    #lblInfo2.grid(column=0,row=2,padx=5,pady=5)
    f.write("\n--------Fecha del analisis : " + fecha2 + " Anomalías: " + leng2)
    f.close()
    img_conv = Image.fromarray(haber)
    
    return leng2




def sobel():
    global image
    gray = image
    flotante = gray.astype(float)
    ker1 = np.array([-0.5,0,0.5])
    ker2 = np.array([[-0.5],[0],[0.5]])
    DX = cv2.filter2D(flotante,-1,ker1)
    DY = cv2.filter2D(flotante,-1,ker2)
    magx = DX**2+DY**2
    magx = np.sqrt(magx)
    magx = magx/np.max(magx)
    maska = np.where(magx>0.1,255,0)
    maska = np.uint8(maska)
    cv2.imshow('Sobel',maska)

def canny():
    global image
    bordes = cv2.Canny(image,30,90)
    ret, binaria = cv2.threshold(bordes,50,255,cv2.THRESH_BINARY_INV)
    contornos,jerarquia = cv2.findContours(binaria, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bordecannenot = cv2.bitwise_not(bordes)
    cv2.imshow('contornos',binaria)

def transformada():
    global image
    Nf = 512
    Nc = 512
    imagentrans = cv2.resize(image,(Nc,Nf))
    flotante = np.float64(imagentrans)
    fu = np.fft.fft2(flotante)
    fu = np.fft.fftshift(fu)
    abs = np.abs(fu)
    log = 20*np.log10(abs)
    B1 = np.arange(-Nf/2+1,Nf/2+1,1)
    B2 = np.arange(-Nc/2+1,Nc/2+1,1)
    [X,Y] = np.meshgrid(B1,B2)
    P = np.sqrt(X**2+Y**2)
    P = P/np.max(P)
    corte = 0.05
    jkl = np.zeros((Nf,Nc))
    for i in range(Nf):
        for u in range(Nc):
            if(P[i,u]<corte):
                jkl[i,u] = 1
