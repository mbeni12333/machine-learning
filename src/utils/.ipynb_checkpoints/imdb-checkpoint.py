import os
import numpy as np
import pickle as pk

DATA_PATH = os.path.join(os.path.abspath("../datasets"), "imdb");


def load_data(data_path=DATA_PATH):
    """
        Load housing data from csv into pandas dataframe
        
        data_path: path to the dataset
        
        return: pandas.DataFrame
    """
    with open (os.path.join(DATA_PATH, "imdb_extrait.pkl"),"rb") as file:
        
        [data , id2titles , fields ]= pk.load(file)
    
    
    datax = data [: ,:33]
    datay = np.array([1 if x [33] >6.5 else -1 for x in data ])
    
    return datax, datay, id2titles, fields
    
    