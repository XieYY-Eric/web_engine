import re
import mmap
import pickle
from nltk import tokenize
from main import normalize

token_pos = {}      # if indexing the index, this will hold
                    # the positions of each token in index_table.txt

# this function uses token_pos to find each token in index_table and return is as str
def find_token_using_token_pos(token):
    pos = token_pos[token]  # get position of token in index_table.txt
    f = open("index_table.txt", "r")
    f.seek(pos)
    curr_token = ""
    while True:  # read entire token from memory
        line = f.readline()
        curr_token += str(line)
        if line.find(']') != -1:
            f.close()
            break
    return curr_token

# this function uses normal file operations to find each token in index_table and return is as str
def find_token(token):
    curr_token = ""
    f = open("./data/index_table.txt","rb")  # get tokens from cache
    while True:  # read entire token from memory
        line = f.readline()
        if line[:len(token)].find(token.encode('utf-8')) != -1:
            curr_token += str(line)
            while line.find(']'.encode('utf-8')) != -1:
                curr_token += str(line)
                break
            break
    f.close()
    return curr_token

def get_token_pos():
    with open("./data/tokens_DEV_cache.p","rb") as f:   # get tokens from cache
        data = pickle.load(f)
    for token in data:      # get position of each token in index_table.txt
        with open(r'./data/index_table.txt', 'rb', 0) as file:
            s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            pos = s.find(token.encode('utf-8'))
        token_pos.update(token=pos)       # add token and its position to token_positions{}


# get_token_pos()   # if using get_token_pos() to get token positions
# get query tokens
print("Enter query : ")
query = str(input())
query_tokens = tokenize.word_tokenize(query)
query_tokens = normalize(query_tokens)
for token in query_tokens:
    print(find_token(token))