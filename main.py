import os
import json
import nltk
from nltk.stem import PorterStemmer
import pickle

### Global variables,
DATA_PATH = "./data/DEV"
PRE_TOKEN_DATA_PATH = "./data/tokens_DEV_cache.p"
PRE_FILENAME_DATA_PATH = "./data/filenames_DEV_cache.p"
USE_CACHE = False  # set to true to load the pre_computed token
PS = PorterStemmer()
###

def get_all_file_names(data_path=DATA_PATH):
    """
    get all the file names from a data_path directory
    return as list
    """
    if USE_CACHE:
        print("Getting file names from cache...")
        data = read_data(PRE_FILENAME_DATA_PATH)
        print(f"Total number of file read: {len(data)}")
        return data

    print("Generating file names...")
    file_names = []
    for root, dirs, files in os.walk(DATA_PATH):
        for file in files:
            file_names.append(os.path.join(root, file))
        print(f"Found {len(files)} file under {root} directory")
    print(f"Total number of file found: {len(file_names)}")
    store_data(file_names, PRE_FILENAME_DATA_PATH)
    return file_names


def get_token_from_file(file_name):
    """
    get all the token from one file, return as a set
    """
    with open(file_name) as f:
        data = json.load(f)
        content = data["content"]
        tokens = nltk.tokenize.word_tokenize(content)
        filtered_tokens = filter(lambda word: word.isalnum(), tokens)
        filtered_tokens = list(filtered_tokens)
        for t in range(len(filtered_tokens)):
            filtered_tokens[t] = filtered_tokens[t].lower()
            filtered_tokens[t] = PS.stem(filtered_tokens[t])
        return set(filtered_tokens)
    
    
def get_all_tokens(dataset):
    """
    get all tokens from the dataset, return as a list
    """
    if USE_CACHE:
        print("Getting tokens from cache...")
        return read_data(PRE_TOKEN_DATA_PATH) 

    print("generating tokens...")
    tokens = set()
    size = len(dataset)
    print_every = 500 #print every 500 files
    for i,filename in enumerate(dataset):
        tokens = tokens.union(get_token_from_file(filename))
        if (i+1)%print_every == 0:
            print(f"{i+1} files completed, {100*(i+1)/size:.2f}%")
    print(f"token generated completed with size {len(tokens)}")
    tokens = list(tokens)
    store_data(tokens,PRE_TOKEN_DATA_PATH)
    return tokens

def store_data(data,filename=PRE_TOKEN_DATA_PATH):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
    print(f"writting data to {filename}")

def read_data(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)

def main():
    dataset = get_all_file_names(DATA_PATH)
    tokens = get_all_tokens(dataset)
    
    
    


if __name__ == "__main__":
    main()
