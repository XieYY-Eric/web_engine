import re
import mmap
import pickle
from nltk import tokenize
import time
import os
import util
import json
import math

class Search_engine:
    def __init__(self,index_table,lookup_table) -> None:
        self.postion_lookup_table = lookup_table
        self.all_indexed_url = util.read_data(util.PRE_INDEXED_URL_PATH)
        self.number_of_url = len(self.all_indexed_url)
        self.number_of_token = len(lookup_table)
        self.index_table = index_table
        self.max_cache_size = 1000 #1000 token cache
        self.cache = {}
        
    def find_posting_using_token_pos(self,postion_lookup_table,token,file):
        # this function uses token_pos to find each token in index_table and return is as str
        hashing_begin = time.time()
        if token not in postion_lookup_table:
            return []
        position = postion_lookup_table[token]
        hashing_end = time.time()
        # print(f"Hashing time {hashing_end-hashing_begin:.3f}")
        seek_begin = time.time()
        file.seek(position)
        seek_end = time.time()
        # print(f"Seeking time {seek_end-seek_begin:.3f}")
        read_begin = time.time()
        curr_token,posting = file.readline().decode("utf-8").strip().split(":")
        read_end= time.time()
        # print(f"read time {read_end-read_begin:.3f}")
        eval_begin = time.time()
        re_expression = r"\(([^\)]+)\)"
        matches = re.findall(re_expression,posting)
        posting = []
        if len(matches) != 0:
            matches =[  match.split(",") for match in matches]
            posting = [(int(dID),float(tf)) for dID,tf in matches ]
        eval_end= time.time()
        # print(f"eval time {eval_end-eval_begin:.3f}")
        return posting

    def search(self,query):
        regex_expression = r"[a-zA-Z\d]+"
        query_tokens = re.findall(regex_expression,query)
        query_tokens = util.normalize(query_tokens)
        query_dict = {}
        for token in query_tokens:
            if token in self.cache:
                query_dict[token] = self.cache[token]
            else:
                query_dict[token] = self.find_posting_using_token_pos(self.postion_lookup_table,token,self.index_table)
                if len(self.cache) >=1000:
                    self.cache.clear()
                self.cache[token] = query_dict[token]
        intersect = get_intersect_posting(query_dict)
        top5_urls = [self.all_indexed_url[docID] for docID, _ in intersect[:5]]
        return top5_urls

    def __str__(self):
        return f"Search_Engine\n\t# of token {self.number_of_token}\n\t# of url {self.number_of_url}"

def get_token_pos_eric(index_table_name,destination_filename):
    begin = time.time()
    position = 0
    position_dict = {}
    f = open(index_table_name,"r",encoding="utf-8")
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

def get_intersect_posting(query_dict):
    """
    query_dict: a dict, token as key, posting as value
    token: a string value
    posting : a list of tuple (DocID, tf_idf_score)

    return : a list of posting (DocID, tf_idf_score)
    """
    start = time.time()
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
    end = time.time()
    print(f"Intersect_posting time: {end-start:.3f}s")
    return ids

def main():
    #get the byte address of each token, look up table
    postion_lookup_table_file_name = "./data/postion_lookup_table.p"
    get_token_pos_eric("./data/index_table.txt",postion_lookup_table_file_name)
    lookup_table =  util.read_data(postion_lookup_table_file_name)

    with open("./data/index_table.txt","rb") as f:
        #pass the file and lookup table to search engine
        search_engine = Search_engine(f,lookup_table)
        print(search_engine)
        while True:
            print("Enter query ('quit' to end): ")
            query = str(input())
            if query == "quit":
                break
            begin = time.time()
            result = search_engine.search(query)
            end = time.time()
            print(f"Top {len(result)} results: {result} \nQuery time {end-begin:.3f}")





if __name__ == "__main__":
    main()



