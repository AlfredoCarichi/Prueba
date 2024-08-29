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
from tkinter import *
from tkinter import filedialog

from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import numpy as np
import matplotlib.path as mplPath
import matplotlib.pyplot as plt
import imutils

def salir():
    exit()

def histograma():
    global image
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
    
def elegir_colonia():
    path_image = filedialog.askopenfilename(filetypes=[
        ("image",".jpg"),
        ('image','.jpeg'),
        ("image",".png")
    ])

    if len(path_image) > 0:
        global image 

        image = cv2.imread(path_image)
        image = imutils.resize(image,height=380)

        imageToShow = imutils.resize(image,width=180)
        imageToShow = cv2.cvtColor(imageToShow,cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imageToShow)
        img = ImageTk.PhotoImage(image=im)
        
        
        
        cv2.imshow('Imagen Ingresada',image)
        lblInputImage.configure(image=img)
        lblInputImage.image = img
        
        lblInfo1 = Label(root,text="Imagen ingresada. ¿Todo correcto?")
        lblInfo1.grid(column=0,row=1,padx=5,pady=5)
        
def deteccion():
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
    
    global image
    model = YOLO("bestArboles.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.05, show_labels=False)[0]
    resultados = model.predict(imagen, imgsz = 640, conf = 0.01)
    detections = sv.Detections.from_ultralytics(result)
    alta = detections[detections.confidence > 0.05]
    print(resultados)
    leng = len(resultados)
    print(leng)
    anotaciones = resultados[0].plot()
    haber = imutils.resize(anotaciones,width=640)
    cv2.imshow("Resutados de la deteccion", haber)
    leng = len(alta)
    leng2 = str(leng)
    print("------------------------------------")
    print("------------------------------------")
    print(leng2 + " Arboles detectados por el algoritmo")
    haber = imutils.resize(anotaciones,width=1024)
    #cv2.imshow("Resultados " + fecha, haber)
    cv2.imwrite(filename, haber)
    lblInfo2 = Label(root,text="Arboles detectados: " + leng2)
    lblInfo2.grid(column=0,row=2,padx=5,pady=5)
    f.write("\n--------Fecha del analisis : " + fecha2 + " Arboles detectados: " + leng2)
    f.close()

    
image = None

root = Tk()
root.title('Version 1.0')
root.config(cursor='plus')

lblInputImage = Label(root)
lblInputImage.grid(column=0,row=2)

lblInfo2 = Label(root,text="Seleccione el algoritmo de bordes",width=25)
lblInfo2.grid(column=0,row=3,padx=5,pady=5)

selected = IntVar()

rad1 = Radiobutton(root,text="YoloV5", width=25,value=1,variable=selected, command=deteccion)
rad2 = Radiobutton(root,text="YoloV7", width=25,value=2,variable=selected)
#rad3 = Radiobutton(root,text="YoloV5", width=25,value=3,variable=selected, command=deteccion2)
rad4 = Radiobutton(root,text="Mostrar histograma", width=25,value=4,variable=selected, command=histograma)
boton = Button(root,text='Deseo salir y cerrar todo',width=25,command=salir)   

rad1.grid(column=0,row=4)
#rad2.grid(column=0,row=5)
#rad3.grid(column=0,row=6)
rad4.grid(column=0,row=3)
boton.grid(column=0,row=7)

btn = Button(root,text='Elegir una imagen', width=25,command=elegir_colonia)
btn.grid(column=0,row=0,padx=5,pady=5)
root.mainloop()

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

    jkl = 1-jkl
    Guv = fu*jkl
    Guvabs = np.abs(Guv)
    Guvabs = np.uint8(255*Guvabs/np.max(Guvabs))


    gxy  = np.fft.ifft2(Guv)
    gxy = np.abs(gxy)
    gxy = np.uint8(gxy)
    cv2.imshow("Filtro circular alto",np.uint8(255*jkl))
    cv2.imshow("Espectro de frecuencia",Guvabs)
    cv2.imshow('Imagen filtrada' ,gxy)


#print("Before saving image:")   
#print(os.listdir(directory))



model = YOLO("bestArboles.pt")
imagen = cv2.imread("prueba2.png")



resultados = model.predict(imagen, imgsz = 640, conf = 0.05)










cv2.waitKey(0)
cv2.destroyAllWindows()


#pip uninstall opencv-python
#pip install opencv-python