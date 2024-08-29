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
from PIL import Image
from functions import *

def main():
    
    st.header('Aplicación para la detección de anomalías de NDVI')
    st.markdown('Esta aplicación esta desarrollada por SEDAGRO Para la detección de las anomalías en los valores del NDVI se utiliza una red neuronal convolucional la cual fue entrenada con una GPU RTX 4060, 200 imágenes de entrenamiento y 40 de validación. Se hicieron 100 épocas las cuales dieron como resultado un archivo PyTorch el cual es el que hace las detecciones. Todo fue desarrollado en el lenguaje Python.')
    file_uploader = st.file_uploader('Sube tu imagen en los siguientes formatos: ', type=['jpg', 'png'])

    if file_uploader is not None:
        image = Image.open(file_uploader)
        print(image)
        image_cv2 = np.array(image)

        st.image(image)
        datos = deteccion(image)
        st.markdown('La paleta de colores original fue modificada para poder observar las detecciones de mejor manera. Valores azules representan valores bajos de NDVI')
        st.image(deteccion((image)))

        st.markdown(deteccion2((image)) + ' anomalías de NDVI detectadas')

        #st.image(histograma(image_cv2))


if __name__ == "__main__":
    main()