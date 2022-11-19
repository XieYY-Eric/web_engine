import re
import mmap
import pickle
from nltk import tokenize
import time
import os
import util
import json


# this function uses token_pos to find each token in index_table and return is as str
def find_posting_using_token_pos(postion_lookup_table,token):
    if token not in postion_lookup_table:
        return []
    position = postion_lookup_table[token]
    with open("./data/index_table.txt", "rb") as f:
        f.seek(position)
        curr_token,posting = f.readline().decode("utf-8").strip().split(":")
        return eval(posting)


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
    util.store_data(position_dict,destination_filename)
    
def get_file_counts():
    file_counts = {}
    f = open("./data/index_table.txt","r")
    for line in f:
        token,postings = line.strip().split(":")
        temp = postings.strip('][').split(', ')
        i = 0
        while i < len(temp):
            doc_id = temp[i].strip(')(')
            count = temp[i+1].strip(')(')
            print(doc_id,end=' : ')
            print(count)
            i += 2
            if doc_id not in file_counts:
                file_counts.update({doc_id:count})
            else:
                curr_count = file_counts[doc_id]
                file_counts.update({doc_id:curr_count+count})
    f.close()
    util.store_data(file_counts, './data/file_counts.p')

def main():
    postion_lookup_table_file_name = "./data/postion_lookup_table.p"
    if not os.path.exists(postion_lookup_table_file_name):
        get_token_pos_eric("./data/index_table.txt",postion_lookup_table_file_name)   # if using get_token_pos() to get token positions
    postion_lookup_table = util.read_data("./data/postion_lookup_table.p")
    all_file_names = util.read_data("./data/filenames_DEV_cache.p")

    #get_file_counts()  # get number of tokens in each file, stores them in file_counts.p
    while True:
        # get query tokens
        print("Enter query ('quit' to end): ")
        query = str(input())
        if query == "quit":
            break
        ##query_tokens = tokenize.word_tokenize(query) [ERIC] since we used regex to tokenize the sentence, we should use regex here as well
        begin = time.time()
        regex_expression = r"[a-zA-Z\d]+"
        query_tokens = re.findall(regex_expression,query)
        query_tokens = util.normalize(query_tokens)
        query_dict = {}
        for token in query_tokens:
            query_dict[token] = find_posting_using_token_pos(postion_lookup_table,token)
        intersect = get_intersect_posting(query_dict)
        
        top5_document = [all_file_names[docID] for docID, _ in intersect]
        urls = []
        for document in top5_document:
            with open(document,"r") as f:
                data = json.load(f)
                urls.append(data["url"])
        end = time.time()
        print(f"Top {len(urls)} results: {urls} Query time {end-begin:.3f}")


def get_intersect_posting(query_dict):
    """
    query_dict: a dict, token as key, posting as value
    token: a string value
    posting : a list of tuple (DocID, frequency)

    return : a list of posting (DocID, frequency)
    """
    intersect = [(1,2),(3,2),(4,2),(5,4),(22,6)]
    ##### step1, sort the query term from smallest posting to highest posting
    tokens = list(query_dict.keys())
    tokens.sort(key=lambda x: len(query_dict[x]))

    ids = []
    combined = query_dict[tokens[0]]
    #### step2, get intersect from each pair one by one
    for token in tokens:
        if token != tokens[0]: 
            postings = dict(ids)
            combined = [(x,y+postings[x]) for x,y in query_dict[token] if x in postings.keys()]
        ids = combined
        combined = []
    ids.sort(key = lambda y: -y[1])
    return ids[0:5]

if __name__ == "__main__":
    main()



