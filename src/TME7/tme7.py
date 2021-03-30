import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import patches
from skimage.transform import resize,rescale
from skimage.color import rgb2gray
from skimage.io import imread
MARKLAND_DIR = Path("FDDB-folds/")
IMAGES_DIR = Path("imgs") 


def load_all_files():
    files = MARKLAND_DIR.glob("*ellipseList*")
    data = dict()
    for f in files:
        data.update(read_markland_file(f))
    return data

def read_markland_file(fn):
    data = dict()
    with open(fn,"r") as f:
        txt = f.read().split("\n")
    i = 0
    while i<len(txt):
        if len(txt[i])<1:
            i+=1
            continue
        cur_path = txt[i].strip()+".jpg"
        nb = int(txt[i+1])
        i+=2
        marks = [[int(x) for x in ellipse2rec([float(txt[i+k].split(" ")[j]) for j in range(5)])] for k in range(nb)]
        img = rgb2gray(imread(IMAGES_DIR / cur_path))
        if img is not None:
            data[cur_path] = {"img":img, "marks": marks}
        i += nb 
    return data

def ellipse2rec(bounding):
    """ Transforme les marquages elliptiques en boungind box """
    a,b =bounding[0], bounding[1]
    angle = bounding[2]
    x,y = bounding[3], bounding[4]
    dx, dy = np.sqrt((a*np.cos(angle))**2+(b*np.sin(angle))**2), np.sqrt((a*np.sin(angle))**2+(b*np.cos(angle))**2)
    return y-dy,x-dx,2*dy,2*dx


def show_img(img,marks=None):
    fix,ax = plt.subplots()
    ax.imshow(img,plt.get_cmap("gray"))
    if marks is not None:
        rect = patches.Rectangle((marks[1],marks[0]),marks[3],marks[2],linewidth=1,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    

def integral_image(img):
    return img.cumsum(axis=1).cumsum(axis=0)





if __name__=="__main__":
    data = load_all_files()
    