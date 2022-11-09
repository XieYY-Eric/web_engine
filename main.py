import os
import json
import nltk
import pickle


### Global variables, 
DATA_PATH = "./data/ANALYST"
PRE_TOKEN_DATA_PATH = "./data/tokens_cache.p"
LOAD_DATA = False #set to true to load the pre_computed token
###

def get_all_file_names(data_path = DATA_PATH):
    """
    get all the file names from a data_path directory
    return as list
    """
    print("Generating file names...")
    file_names = []
    for root,dirs,files in os.walk(DATA_PATH):
        for file in files:
            file_names.append(os.path.join(root,file))
        print(f"Found {len(files)} file under {root} directory")
    print(f"Total number of file found {len(file_names)}")
    return file_names

def get_token_from_file(file_name):
    """
    get all the token from one file, return as a set
    """
    with open(file_name) as f:
        data = json.load(f)
        content = data["content"]
        tokens = nltk.tokenize.word_tokenize(content)
        stop_words = set(nltk.corpus.stopwords.words('english'))
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
        return set(filtered_tokens)

def get_all_tokens(dataset):
    """
    get all tokens from the dataset, return as a set
    """
    print("generating tokens...")
    tokens = set()
    size = len(dataset)
    print_every = 100 #print every 20 files
    for i,filename in enumerate(dataset):
        tokens = tokens.union(get_token_from_file(filename))
        if (i+1)%print_every == 0:
            print(f"{i+1} files completed, {100*(i+1)/size:.2f}%")
    print(f"token generated completed with size {len(tokens)}")
    return tokens

def store_data(data,filename=PRE_TOKEN_DATA_PATH):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
    print(f"writting data to {filename}")

def read_data(filename=PRE_TOKEN_DATA_PATH):
    with open(filename,"rb") as f:
        return pickle.load(f)

def main():
    tokens = []
    if LOAD_DATA:
        tokens = read_data(PRE_TOKEN_DATA_PATH)
    else:
        dataset = get_all_file_names(DATA_PATH)
        tokens = list(get_all_tokens(dataset))
        store_data(tokens,PRE_TOKEN_DATA_PATH)
    
    
    
    


if __name__ == "__main__":
    main()