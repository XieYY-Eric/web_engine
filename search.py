import re
import mmap
import pickle
from nltk import tokenize
from main import normalize
import time
import os

token_pos = { '000th':0 }      # if indexing the index, this will hold
                    # the positions of each token in index_table.txt

# this function uses token_pos to find each token in index_table and return is as str
def find_posting_using_token_pos(postion_lookup_table,token):
    if token not in postion_lookup_table:
        return []
    position = postion_lookup_table[token]
    with open("./data/index_table.txt", "rb") as f:
        f.seek(position)
        curr_token,posting = f.readline().decode("utf-8").strip().split(":")
        return eval(posting)

# # this function uses normal file operations to find each token in index_table and return is as str
# def find_token(token):
#     curr_token = ""
#     f = open("./data/index_table.txt","rb")  # get tokens from cache
#     while True:  # read entire token from memory
#         line = f.readline()
#         if not line:
#             return ''
#         if line[:len(token)].find(token.encode('utf-8')) != -1:
#             curr_token += str(line)
#             while line.find(']'.encode('utf-8')) != -1:
#                 curr_token += str(line)
#                 break
#             break
#     f.close()
#     curr_token = curr_token[2:]
#     curr_token = curr_token[:len(curr_token) - 5]
#     return curr_token

# def get_token_pos():
#     with open("./data/tokens_DEV_cache.p","rb") as f:   # get tokens from cache
#         data = pickle.load(f)
#         data = list(data)
#         data.sort()
#     for token in data[:10]:      # get position of each token in index_table.txt
#         with open(r'./data/index_table_small.txt', 'rb', 0) as file:
#             s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
#             pos = s.find(token.encode('utf-8'))
#         token_pos.update(token=pos)       # add token and its position to token_positions{}

def get_token_pos_eric(index_table_name,destination_filename):
    begin = time.time()
    position = 0
    position_dict = {}
    f = open(index_table_name,"r")
    for line in f:
        token,posting = line.strip().split(":")
        position_dict[token] = position
        position += len(token)
        position += len(posting)
        position += 3
    f.close()
    end = time.time()
    print(f"finished creating token_map, time:{end-begin:.3f}")
    with open(destination_filename,"wb") as f:
        pickle.dump(position_dict,f)
    


get_token_pos_eric("./data/index_table.txt","./data/postion_lookup_table.p")   # if using get_token_pos() to get token positions
postion_lookup_table = None
with open("./data/postion_lookup_table.p","rb") as f:
    postion_lookup_table = pickle.load(f)


# get query tokens
print("Enter query : ")
query = str(input())
##
##query_tokens = tokenize.word_tokenize(query) [ERIC] since we used regex to tokenize the sentence, we should use regex here as well
regex_expression = r"[a-zA-Z\d]+"
query_tokens = re.findall(regex_expression,query)
query_tokens = normalize(query_tokens)
for token in query_tokens:
    print(f"token {token} :-> {find_posting_using_token_pos(postion_lookup_table,token)}")