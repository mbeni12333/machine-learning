# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 09:19:37 2021

@author: Mounib Benimam
"""

import numpy as np
import matplotlib.pyplot as plt
import os


uspsdatatrain = os.path.abspath("../Datasets/USPS/USPS_train.txt")
uspsdatatest = os.path.abspath("../Datasets/USPS/USPS_test.txt")


def load_usps(mode="train"):
    """
    

    Parameters
    ----------
    mode : TYPE, optional
        DESCRIPTION. The default is "train".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    
    
    fn = uspsdatatrain if mode == "train" else uspsdatatest
    
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    """
    

    Parameters
    ----------
    l : TYPE
        DESCRIPTION.
    datax : TYPE
        DESCRIPTION.
    datay : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")