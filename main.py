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

### Global variables, 
DATA_PATH = "./data/DEV"
PRE_TOKEN_DATA_PATH = "./data/tokens_DEV_cache.p"
PRE_FILENAME_DATA_PATH = "./data/filenames_DEV_cache.p" 
INDEX_TABLE_PREFIX="./data/Index_tables/"
USE_CACHE = True 
MAX_WORD = 50000
MIN_WORD = 10
###



##### Private/Helper/Inner function ---- usually called by other important function
def is_html(content):
    html_tag = [r"<!DOCTYPE html",r"<!DOCTYPE HTML"]
    for tag in html_tag:
        if tag in content:
            return True
    return False #expecting the top few string are html_tag

def can_be_index(content):
    try:
        if not is_html(content):
            print(content[:100])
            return False
        content = BeautifulSoup(content).get_text()
        if len(content) < MIN_WORD or len(content) > MAX_WORD:
            print("word count",len(content))
            return False
        return True
    except:
        print("Exception")
        return False

def get_token_from_file(file_name):
    """
    get all the token from one file, return as a tuple (status_code, result)
    """
    with open(file_name) as f:
        try:
            data = json.load(f)
            content = data["content"] #could be string or html string
            if not can_be_index(content):
                print(f"name {file_name}")
                input()
                return (-1,set())
            soup = BeautifulSoup(content, "html.parser")
            content = ""
            for section in soup.find_all('p'):
                content += (section.get_text() + " ") 
            tokens = nltk.tokenize.word_tokenize(content)
            filtered_tokens = normalize(filtered_tokens)
            return (1,set(filtered_tokens))
        except Exception as e:
            return (-1,set())

def store_data(data,filename=PRE_TOKEN_DATA_PATH):
    with open(filename,"wb") as f:
        pickle.dump(data,f)
    print(f"writting data to {filename}")

def read_data(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)
#####

##### Public function, ----important one, user can call these
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

def get_all_files_and_tokens(dataset):
    """
    get all tokens from the dataset, return as a list
    """
    if USE_CACHE:
        print("Getting tokens from cache...")
        return read_data(PRE_FILENAME_DATA_PATH),read_data(PRE_TOKEN_DATA_PATH) 
    indexable_files = []
    print("generating tokens...")
    tokens = set()
    size = len(dataset)
    print_every = 500 #print every 500 files
    begin = time.time()
    end = time.time()
    for i,filename in enumerate(dataset):
        status,tokens_result = get_token_from_file(filename)
        if status == 1:
            tokens = tokens.union(tokens_result)
            indexable_files.append(filename)

        if (i+1)%print_every == 0:
            end = time.time()
            print(f"{i+1} files completed, {100*(i+1)/size:.2f}%  time {(end-begin):.3f}s")
            begin = end

    print(f"number of indexable files: {len(indexable_files)}")
    print(f"token generated completed with size {len(tokens)}")
    tokens = list(tokens)
    # store_data(indexable_files,PRE_FILENAME_DATA_PATH)
    # store_data(tokens,PRE_TOKEN_DATA_PATH)
    return indexable_files,tokens


def normalize(tokenlist):
    #remove punctuation
    filter_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokenlist]
    #remove non-alphanumeric
    tokenlist = filter(lambda token: token.isalnum(), tokenlist)
    #lower
    filter_tokens = [token.lower() for token in tokenlist]
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





def main():
    dataset = get_all_file_names(DATA_PATH)
    # indexablefiles,tokens = get_all_files_and_tokens(dataset)
    # print("before normalizing:",len(tokens),tokens[:20])
    # tokens = normalize(tokens)
    # tokens = unique(tokens)
    # print("after normalizing:",len(tokens),tokens[:20])
    # filekb = os.path.getsize(PRE_FILENAME_DATA_PATH) /1024
    # print("Index is", filekb, "KBs large")

 
    ###indexing pages
    myindexer = indexer.indexer(dataset,1024,INDEX_TABLE_PREFIX,MIN_WORD,MAX_WORD)
    myindexer.index_all_Doc()
    ###create batches of files
    index_table1 = read_data("./data/Index_tables/0.p")
    index_table2 = read_data("./data/Index_tables/1.p")
    if USE_CACHE:
        store_data(myindexer.indexedDocument,"./data/Index_tables/IndexedPages.p")
    # print(index_table1)
    # print("--------------\n")
    # print(index_table2)

    


if __name__ == "__main__":
    main()
