import numpy as np
import matplotlib.pyplot as plt

from src.MLLIB.mltools import plot_data, plot_frontiere, make_grid, gen_arti

def sigmoid(X):
    
    return 1.0/(1.0 + np.exp(-X))

def mse(w,x,y):
    """
    

    Parameters
    ----------
    w : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    m = len(y)
    return 1.0/m * np.linalg.norm(x @ w - y)**2

def mse_grad(w, x, y):
    
    m = len(y)
    y = y.reshape(-1, 1)
    
    dw = 2.0/m * x.T @ (x@w - y)
   
    return dw


def reglog(w, x, y):
    
    y = y.reshape(-1, 1)  
    y_hat = sigmoid(x @ w).reshape(-1, 1)
    
    return -(y.T @ np.log(y_hat + 1e-14) + (1-y).T @ np.log(1-y_hat + 1e-14))

def reglog_grad(w, x, y):
    
    m = len(y)
    y = y.reshape(-1, 1)
    y_hat = sigmoid(x@w).reshape(-1, 1)
    
    return -1.0/m * x.T @ (y - y_hat)

def grad_check(f,f_grad,N=100):
    return 0


def descente_gradient(datax,datay,f_loss,f_grad,eps,max_iters):
    
    Ws = []
    Ls = []
    W = np.random.randn(datax.shape[1], 1)
    W_best = None
    L_best = np.inf
    
    for i in range(max_iters):
        L = np.asscalar(f_loss(W, datax, datay))
        dW = f_grad(W, datax, datay)
        
        W = W - eps*dW
        
        Ws.append(W)
        Ls.append(L)
        
        if(L < L_best):
            L_best = L
            W_best = W
    
    
    return W_best, np.array(Ws), Ls



if __name__=="__main__":
    ## Tirage d'un jeu de données aléatoire avec un bruit de 0.1
    datax, datay = gen_arti(epsilon=0.1)
    ## Fabrication d'une grille de discrétisation pour la visualisation de la fonction de coût
    grid, x, y = make_grid(xmin=-2, xmax=100, ymin=-2, ymax=100, step=100)
    
    #plt.figure()
    ## Visualisation des données et de la frontière de décision pour un vecteur de poids w
    #w  = np.random.randn(datax.shape[1],1)
    #plot_frontiere(datax,lambda x : np.sign(x.dot(w)),step=100)
    #plot_data(datax,datay)

    ## Visualisation de la fonction de coût en 2D
    #plt.figure()
    #plt.contourf(x,y,np.array([mse(w,datax,datay) for w in grid]).reshape(x.shape),levels=50)
    
    
    
    
    ############################## REGLOG ##########################################
    
    W_best, Ws, Ls = descente_gradient(datax, datay, reglog, reglog_grad, 1e-1, 100)
    
    
    plt.figure()
    plt.plot(Ls)
    
    plt.figure()
    plot_frontiere(datax,lambda x : sigmoid(x.dot(W_best))>0.5,step=100)
    plot_data(datax,datay)
    
    plt.figure()
    plt.contourf(x,y,np.array([reglog(W_best, datax, datay) for w in grid]).reshape(x.shape),levels=50)
    plt.colorbar()
    plt.plot(Ws[::20, 0], Ws[::20, 1], marker="x", color="red")
    plt.scatter(Ws[-1, 0], Ws[-1, 1], marker="x", color="white")
    

    
    ################################ MSE #############################################
    
    grid, x, y = make_grid(xmin=-2, xmax=4, ymin=-2, ymax=4, step=100)
    W_best, Ws, Ls = descente_gradient(datax, datay, mse, mse_grad, 1e-2, 100)
    
        
    plt.figure()
    plt.plot(Ls)
   
    
    plt.figure()
    plot_frontiere(datax,lambda x : np.sign(x.dot(W_best)),step=100)
    plot_data(datax,datay)
    
    plt.figure()
    plt.contourf(x,y,np.array([mse(w, datax, datay) for w in grid]).reshape(x.shape),levels=50)
    plt.colorbar()
    plt.plot(Ws[::20, 0], Ws[::20, 1], marker="x", color="red")
    plt.scatter(Ws[-1, 0], Ws[-1, 1], marker="x", color="white")