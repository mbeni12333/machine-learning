import os
import urllib
import tarfile
import pandas as pd



DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
        This function downloads California housing datasets
        
        housing_url: url to fetch the dataset from
        housing_path: path to store the dataset to
    """
    
    # create the folder if don't exist
    os.makedirs(housing_path, exist_ok=True)
    
    # the path to the compressed version
    tgz_path = os.path.join(housing_path, "housing.tgz")
    
    # Download the data 
    urllib.request.urlretrieve(housing_url, tgz_path)
    
    # Extract the data
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    """
        Load housing data from csv into pandas dataframe
        
        housing_path: path to the dataset
        
        return: pandas.DataFrame
    """
    
    # file path
    csv_path = os.path.join(housing_path, "housing.csv")
    
    #return the dataframe
    return pd.read_csv(csv_path)
    
    