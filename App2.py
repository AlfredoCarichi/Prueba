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

#Librerias del segundo proceso
import rasterio
import geopandas as gpd
import numpy as np
from matplotlib import pyplot as plt
import cv2
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.plot import show
import os
from shapely.geometry import box
from shapely.ops import unary_union
import json
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
import shapely
import math
import pandas as pd
from skimage import io
from skimage.io import imsave
from sklearn import model_selection
import os
import shutil
import json
import ast
import numpy as np
from tqdm import tqdm
import pandas as  pd
import seaborn as sns
import fastai.vision as vision
import xml.etree.ElementTree as ET
import glob
from shapely.geometry import Polygon
from shutil import rmtree



def shapefile():

    if not os.path.isdir('Resultados'):
        os.mkdir('Resultados')
        
    if not os.path.isdir('Predict'):
        os.mkdir('Predict')

    if not os.path.isdir('Predict_jpg'):
        os.mkdir('Predict_jpg')
    
    if not os.path.isdir('runs'):
        os.mkdir('runs')
    rmtree("Predict")
    rmtree("Predict_jpg")
    rmtree("Resultados")
    rmtree("runs")
    global ras
    path_img = ras
    src_img = rasterio.open(path_img)
    img = src_img.read()
    print(img.shape)
    img = img.transpose([1,2,0])
    plt.figure(figsize=[16,16])
    plt.imshow(img)
    plt.axis('off')
    
    #Dvividir el ortomosaico en imagenes mas pequeñas (1500 X 1500 pixeles)
    if not os.path.isdir('Resultados'):
        os.mkdir('Resultados')
        
    if not os.path.isdir('Predict'):
        os.mkdir('Predict')

    if not os.path.isdir('Predict_jpg'):
        os.mkdir('Predict_jpg')
    
    if not os.path.isdir('runs'):
        os.mkdir('runs')
    
    qtd = 0
    out_meta = src_img.meta.copy()
    for n in range((src_img.meta['width']//1500)):
        for m in range((src_img.meta['height']//1500)):
            x = (n*1500)
            y = (m*1500)
            window = Window(x,y,1500,1500)
            win_transform = src_img.window_transform(window)
            arr_win = src_img.read(window=window)
            arr_win = arr_win[0:3,:,:]
            if (arr_win.max() != 0) and (arr_win.shape[1] == 1500) and (arr_win.shape[2] == 1500):
                qtd = qtd + 1
                path_exp = 'Predict/img_' + str(qtd) + '.tif'
                out_meta.update({"driver": "GTiff","height": 1500,"width": 1500, "transform":win_transform})
                with rasterio.open(path_exp, 'w', **out_meta) as dst:
                    for i, layer in enumerate(arr_win, start=1):
                        dst.write_band(i, layer.reshape(-1, layer.shape[-1]))
                del arr_win
    print(qtd)
    
    #Convertimos las imagenes de formato tiff a jpg ya que es el formato que utiliza Yolo
    if not os.path.isdir('Predict_jpg'):
        os.mkdir('Predict_jpg')
        
    path_data_pred = 'Predict_jpg'
    imgs_to_pred = os.listdir('Predict')
    
    for images in imgs_to_pred:
        src = rasterio.open('Predict/' + images)
        raster = src.read()
        raster = raster.transpose([1,2,0])
        raster = raster[:,:,0:3]
        imsave(os.path.join(path_data_pred,images.split('.')[0] + '.jpg'), raster)
        
    #Importamos Yolo y utilizamos un modelo OBB preentrenado
    from ultralytics import YOLO
    model = YOLO('Models/best.pt')
    model.predict('Predict_jpg', save=True,save_txt=True, show_conf=False, show_labels=False, imgsz=640, conf=0.05)
    from IPython.display import Image, display
    for images in glob.glob('runs/obb/predict/*.jpg')[316:320]:
        display(Image(filename=images))
    
    #Se agregan las coordenadas a los Bounding Boxes
    ls_poly = []
    ls_class = []
    
    imgs_to_pred = [f for f in os.listdir('runs/obb/predict/labels/') if f.endswith('.txt')]
    for images in imgs_to_pred:
        filename = images.split('.')[0]
        src = rasterio.open('Predict/' + filename + '.tif')
        path = f'runs/obb/predict/labels/'+filename+'.txt'
        cols = ['class', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        df = pd.read_csv(path, sep=" ", header=None)
        df.columns = cols
        df['x1'] = np.round(df['x1'] * 1500)
        df['y1'] = np.round(df['y1'] * 1500)
        df['x2'] = np.round(df['x2'] * 1500)
        df['y2'] = np.round(df['y2'] * 1500)
        df['x3'] = np.round(df['x3'] * 1500)
        df['y3'] = np.round(df['y3'] * 1500)
        df['x4'] = np.round(df['x4'] * 1500)
        df['y4'] = np.round(df['y4'] * 1500)

        for i,row in df.iterrows():
            xs1, ys1 = rasterio.transform.xy(src.transform, row['y1'], row['x1'])
            xs2, ys2 = rasterio.transform.xy(src.transform, row['y2'], row['x2'])
            xs3, ys3 = rasterio.transform.xy(src.transform, row['y3'], row['x3'])
            xs4, ys4 = rasterio.transform.xy(src.transform, row['y4'], row['x4'])

            ls_poly.append(Polygon([[xs1, ys1], [xs2,ys2], [xs3, ys3], [xs4,ys4]]))
            ls_class.append(row['class'])
            
    #Importamos los resultados a un DataFrame
    gdf = gpd.GeoDataFrame(ls_class, geometry=ls_poly, crs=src.crs)
    gdf.rename(columns={0:'class'}, inplace=True)
    print(gdf)
    gdf.to_file('Resultados/Conteo_Arboles.json')
    
    
def shapefile2():
    rmtree("Predict")
    rmtree("Predict_jpg")
    rmtree("Resultados")
    rmtree("runs")
    global ras
    path_img = ras
    src_img = rasterio.open(path_img)
    img = src_img.read()
    print(img.shape)
    img = img.transpose([1,2,0])
    plt.figure(figsize=[16,16])
    plt.imshow(img)
    plt.axis('off')
    
    #Dvividir el ortomosaico en imagenes mas pequeñas (1500 X 1500 pixeles)
    if not os.path.isdir('Resultados'):
        os.mkdir('Resultados')
        
    if not os.path.isdir('Predict'):
        os.mkdir('Predict')
    
    qtd = 0
    out_meta = src_img.meta.copy()
    for n in range((src_img.meta['width']//1500)):
        for m in range((src_img.meta['height']//1500)):
            x = (n*1500)
            y = (m*1500)
            window = Window(x,y,1500,1500)
            win_transform = src_img.window_transform(window)
            arr_win = src_img.read(window=window)
            arr_win = arr_win[0:3,:,:]
            if (arr_win.max() != 0) and (arr_win.shape[1] == 1500) and (arr_win.shape[2] == 1500):
                qtd = qtd + 1
                path_exp = 'Predict/img_' + str(qtd) + '.tif'
                out_meta.update({"driver": "GTiff","height": 1500,"width": 1500, "transform":win_transform})
                with rasterio.open(path_exp, 'w', **out_meta) as dst:
                    for i, layer in enumerate(arr_win, start=1):
                        dst.write_band(i, layer.reshape(-1, layer.shape[-1]))
                del arr_win
    print(qtd)
    
    #Convertimos las imagenes de formato tiff a jpg ya que es el formato que utiliza Yolo
    if not os.path.isdir('Predict_jpg'):
        os.mkdir('Predict_jpg')
        
    path_data_pred = 'Predict_jpg'
    imgs_to_pred = os.listdir('Predict')
    
    for images in imgs_to_pred:
        src = rasterio.open('Predict/' + images)
        raster = src.read()
        raster = raster.transpose([1,2,0])
        raster = raster[:,:,0:3]
        imsave(os.path.join(path_data_pred,images.split('.')[0] + '.jpg'), raster)
        
    #Importamos Yolo y utilizamos un modelo OBB preentrenado
    from ultralytics import YOLO
    model = YOLO('Models/yolov8x-obb.pt')
    model.predict('Predict_jpg', save=True,save_txt=True, show_conf=False, show_labels=False, imgsz=640, conf=0.1)
    from IPython.display import Image, display
    for images in glob.glob('runs/obb/predict/*.jpg')[316:320]:
        display(Image(filename=images))
    
    #Se agregan las coordenadas a los Bounding Boxes
    ls_poly = []
    ls_class = []
    
    imgs_to_pred = [f for f in os.listdir('runs/obb/predict/labels/') if f.endswith('.txt')]
    for images in imgs_to_pred:
        filename = images.split('.')[0]
        src = rasterio.open('Predict/' + filename + '.tif')
        path = f'runs/obb/predict/labels/'+filename+'.txt'
        cols = ['class', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
        df = pd.read_csv(path, sep=" ", header=None)
        df.columns = cols
        df['x1'] = np.round(df['x1'] * 1500)
        df['y1'] = np.round(df['y1'] * 1500)
        df['x2'] = np.round(df['x2'] * 1500)
        df['y2'] = np.round(df['y2'] * 1500)
        df['x3'] = np.round(df['x3'] * 1500)
        df['y3'] = np.round(df['y3'] * 1500)
        df['x4'] = np.round(df['x4'] * 1500)
        df['y4'] = np.round(df['y4'] * 1500)

        for i,row in df.iterrows():
            xs1, ys1 = rasterio.transform.xy(src.transform, row['y1'], row['x1'])
            xs2, ys2 = rasterio.transform.xy(src.transform, row['y2'], row['x2'])
            xs3, ys3 = rasterio.transform.xy(src.transform, row['y3'], row['x3'])
            xs4, ys4 = rasterio.transform.xy(src.transform, row['y4'], row['x4'])

            ls_poly.append(Polygon([[xs1, ys1], [xs2,ys2], [xs3, ys3], [xs4,ys4]]))
            ls_class.append(row['class'])
            
    #Importamos los resultados a un DataFrame
    gdf = gpd.GeoDataFrame(ls_class, geometry=ls_poly, crs=src.crs)
    gdf.rename(columns={0:'class'}, inplace=True)
    print(gdf)
    gdf.to_file('Resultados/Conteo_Arboles.json')
    
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
def elegir_raster():
    path_raster = filedialog.askopenfilename(filetypes=[
        ("raster",".tif")
    ])

    if len(path_raster) > 0:
     
        global ras 
        print(path_raster)
        ras = path_raster


        
        
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
    model = YOLO("Models/bestNDVI.pt")
    imagen = image
    result = model(imagen,imgsz = 640, conf = 0.1, show_labels=False)[0]
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
ras = None

root = Tk()
root.title('Version 1.0')
root.config(cursor='plus')

lblInputImage = Label(root)
lblInputImage.grid(column=0,row=2)

lblInfo2 = Label(root,text="Seleccione el algoritmo de bordes",width=25)
lblInfo2.grid(column=0,row=3,padx=5,pady=5)

selected = IntVar()

rad1 = Radiobutton(root,text="YoloV5 (JPG, PNG)", width=25,value=1,variable=selected, command=deteccion)
rad2 = Radiobutton(root,text="Raster (TIF)", width=25,value=2,variable=selected, command=shapefile)
rad3 = Radiobutton(root,text="Detección Huertas (TIF)", width=25,value=3,variable=selected, command=shapefile2)
rad4 = Radiobutton(root,text="Mostrar histograma", width=25,value=4,variable=selected, command=histograma)
boton = Button(root,text='Deseo salir y cerrar todo',width=25,command=salir)   

rad1.grid(column=0,row=5)
rad2.grid(column=0,row=6)
rad3.grid(column=0,row=7)
rad4.grid(column=0,row=4)
boton.grid(column=0,row=8)

btn = Button(root,text='Elegir una imagen', width=25,command=elegir_colonia)
btn2 = Button(root,text="Elegir capa en formato raster",width=25,command=elegir_raster)
btn.grid(column=0,row=0,padx=5,pady=5)
btn2.grid(column=0,row=3, pady=5,padx=5)
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






model = YOLO("bestArboles.pt")
imagen = cv2.imread("prueba2.png")



resultados = model.predict(imagen, imgsz = 640, conf = 0.05)










cv2.waitKey(0)
cv2.destroyAllWindows()


#pip uninstall opencv-python
#pip install opencv-python