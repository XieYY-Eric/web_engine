import os
import json
import nltk
from nltk.stem import PorterStemmer
import pickle
import time
from bs4 import BeautifulSoup
import re
import string
import indexer
import util

### Global variables, 
DATA_PATH = "./data/DEV"
PRE_TOKEN_DATA_PATH = "./data/tokens_DEV_cache.p"
PRE_FILENAME_DATA_PATH = "./data/filenames_DEV_cache.p" 
INDEX_TABLE_PREFIX="./data/Index_tables/"
MAX_WORD = 100000
MIN_WORD = 10
USE_CACHE = True
###

def get_all_file_names(data_path=DATA_PATH):
    """
    get all the file names from a data_path directory
    return as list
    """
    print("Generating file names...")
    file_names = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            file_names.append(os.path.join(root, file))
        print(f"Found {len(files)}  file under {root} directory")
    print(f"Total number of  file found: {len(file_names)}")
    return file_names

def normalize(tokenlist):
    # #remove punctuation
    filter_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokenlist]
    #lower
    filter_tokens = [token.lower() for token in filter_tokens]
    #pure number
    filter_tokens = [token for token in filter_tokens if not str.isnumeric(token)]
    #stemming
    #filter_tokens = [PS.stem(token,to_lowercase=True) for token in tokenlist]
    return filter_tokens

def unique(tokenlist):
    #removing duplicate
    unique = set()  
    result = []
    for token in tokenlist:
        if token not in unique:
            result.append(token)
            unique.add(token)
    return result

def store_data(data,filename=PRE_TOKEN_DATA_PATH):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
    print(f"writting data to {filename}")

def read_data(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)

