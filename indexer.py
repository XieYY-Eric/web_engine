from ast import Lambda
from lib2to3.pgen2.tokenize import tokenize
import pickle
from collections import namedtuple
import json
from turtle import towards
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
import nltk
import time
import math
import regex as re
import util


def getTokens(content):
    soup = BeautifulSoup(content)
    data = ""
    for section in soup.find_all('p'):
        text = section.get_text()
        data += (text + " ")
    regex_expression = r"[a-zA-Z\d]+"
    tokens = re.findall(regex_expression,data)
    return tokens

def is_html(content):
    html_tag = [r"<!DOCTYPE html",r"<!DOCTYPE HTML",r"html"]
    for tag in html_tag:
        if tag in content[:100]:
            return True
    return False #expecting the top few string are html_tag

      
Pair = namedtuple("Pair", ["DocID", "Count"])
class indexer:
    #IMPORTANT: CALL INDEXER CLOSE BEFORE TERMINATING THE PROGRAM!!!
    #TO-DO M2: add tf-idf score. Run unique tokens through and take the last in the array and the second of the pair and divide by word count
    def __init__(self, allDocuments,batch_size,index_file_prefix,min_word,max_word):
        '''make the indexer add the inverted index filepaths and the indexer of that'''
        self.allDocuments = allDocuments
        self.fileId = 0
        self._partial_index_file_prefix = index_file_prefix
        self._batch_size = batch_size
        self.min_word = min_word
        self.max_word = max_word
        self.indexedDocument = []
        self.unique_tokens = set()
        self.logfile = None

    def index_DocBatch(self,documents_batch):
        index_table = {}
        for d in documents_batch:
            try:
                file = open(d,"r")
                data = json.load(file)
                data = data["content"]
                ##process the string
                if is_html(data):
                    tokens = getTokens(data)
                    tokens = util.normalize(tokens)
                    if len(tokens) > self.min_word and len(tokens) < self.max_word:
                        fredict = nltk.FreqDist(tokens)
                        for k,v in fredict.items():
                            p = Pair(self.fileId,v)
                            if k not in index_table:
                                index_table[k] = [p]
                            else:
                                index_table[k].append(p)
                        self.indexedDocument.append(d)
                        self.fileId += 1
                else:
                    self.logfile.write(f"filename {d} content {data[:20]}\n")
                file.close()
            except Exception as e:
                print(f"filename {d} Exception {e}")
        return index_table
    
    def batch_of_documents(self,documents,size):
        num_of_files = len(documents)
        for i in range(0,num_of_files,size):
            yield documents[i:min(i+size,num_of_files)]

    def index_all_Doc(self):
        self.logfile = open("./data/LogFile.txt","w")
        begin = time.time()
        end = time.time()
        number_of_batch = math.ceil(len(self.allDocuments)/self._batch_size)
        for i,batch in enumerate(self.batch_of_documents(self.allDocuments,self._batch_size)):
            file_to_store = self._partial_index_file_prefix+str(i)+".p"
            index_table = self.index_DocBatch(batch)
            self.unique_tokens = self.unique_tokens.union(set(index_table.keys()))
            util.store_data(index_table,file_to_store)
            end = time.time()
            print(f"Batch {i+1}/{number_of_batch} completed {(end-begin):.3f}s")
            begin = end
        print(f"number of pages indexed {len(self.indexedDocument)}")
        util.store_data(self.indexedDocument,"./data/filenames_DEV_cache.p" )
        util.store_data(self.unique_tokens,"./data/tokens_DEV_cache.p" )
        self.logfile.close()
    


    


              


