import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from src.MLLIB.mltools import plot_data, plot_frontiere, make_grid, gen_arti
from src.Datasets.USPS.loader import load_usps, get_usps, show_usps
    

def perceptron_loss():
    return 0

def perceptron_grad():
    return 0

class Lineaire(object):
    def __init__(self,loss=perceptron_loss,loss_g=perceptron_grad,max_iter=100,eps=0.01):
        self.max_iter, self.eps = max_iter,eps
        self.w = None
        self.loss,self.loss_g = loss,loss_g
        
    def fit(self,datax,datay):
        pass

    def predict(self,datax):
        pass

    def score(self,datax,datay):
        pass



if __name__ =="__main__":
    alltrainx,alltrainy = load_usps("train")
    alltestx,alltesty = load_usps("test")
    neg, pos = 5, 6

    datax,datay = get_usps([neg,pos],alltrainx,alltrainy)
    testx,testy = get_usps([neg,pos],alltestx,alltesty)
    
    show_usps(datax[0])
