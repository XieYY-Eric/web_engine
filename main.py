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





##### Private/Helper/Inner function ---- usually called by other important function


#####

##### Public function, ----important one, user can call these

def format_file(tokens,partial_tables,filename):
    #get all the tokens
    f = open(filename,"w")
    for token in tokens:
        if token in partial_tables:
            data =[(tuple[0],tuple[1]) for tuple in partial_tables[token]]
            f.write(f"{token}:{data}\n")
        else:
            f.write(f"{token}:[]\n")
    f.close()

def format_all_files(number_of_files):
    tokens = list(util.read_data(util.PRE_TOKEN_DATA_PATH))
    tokens.sort()
    for i in range(number_of_files):
        partial_table = util.read_data(util.INDEX_TABLE_PREFIX+str(i)+".p")
        format_file(tokens,partial_table,"./data/Index_tables/"+str(i)+".txt")
        print(f"file {i} formatted completed")

def merge_all_files(number_of_files):
    filenames = ["./data/Index_tables/"+str(i)+".txt" for i in range(number_of_files)]
    files = [open(filename,"r") for filename in filenames]
    next_lines = [files[i].readline() for i in range(number_of_files)]
    line = next_lines[0]
    file_to_write = "./data/index_table.txt"
    f = open(file_to_write,"w")
    print_every = 10000
    count = 0
    while line:
        values = []
        token = ""
        for this_line in next_lines:
            token,value = this_line.strip().split(":")
            values.extend(eval(value))
        f.write(f"{token}:{values}\n")
        count += 1
        if count%print_every == 0:
            print(f"merging token {count}")
        next_lines = [files[i].readline() for i in range(number_of_files)]
        line = next_lines[0]
    f.close()
    closing = [file.close() for file in files]



def main():
    dataset = util.get_all_file_names(util.DATA_PATH)


 
    # #indexing pages
    myindexer = indexer.indexer(dataset,1024,util.INDEX_TABLE_PREFIX,util.MIN_WORD,util.MAX_WORD)
    myindexer.index_all_Doc() #COMMENT out this if you dont wanna computing all over again

    token = util.read_data(util.PRE_TOKEN_DATA_PATH)
    index_file = util.read_data(util.PRE_FILENAME_DATA_PATH)
    print(f"number of tokens {len(token)}\nnumber of files {len(index_file)}")

    #format all 46 partial index table files
    format_all_files(46)
    #merge
    merge_all_files(46)


if __name__ == "__main__":
    main()
