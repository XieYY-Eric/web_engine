from ast import Lambda
from lib2to3.pgen2.tokenize import tokenize
import pickle
from collections import namedtuple
import json
from turtle import towards
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import math
import regex as re
import util



def is_html(content):
    html_tag = "html"
    return html_tag in content[:1000].lower()

def getTokens(content):
    data = ""
    if is_html(content):
        #is html file, index it
        soup = BeautifulSoup(content)
        #### all find paragraghs
        # for section in soup.find_all('p'):
        #     text = section.get_text()
        #     data += (text + " ")
        #### all find text
        data = soup.get_text(separator=" ",strip=True)
    else:
        #non html, treat it as raw page
        data = content
    regex_expression = r"[a-zA-Z\d]+"
    tokens = re.findall(regex_expression,data)
    return tokens



Pair = namedtuple("Pair", ["DocID", "tf"])
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
        self.indexedUrl = []
        self.unique_tokens = set()
        self.logfile = None

    def index_DocBatch(self,documents_batch):
        index_table = {}
        for d in documents_batch:
            try:
                file = open(d,"r")
                data = json.load(file)
                url = data["url"]
                data = data["content"]
                ##process the string, get the token from string
                tokens = getTokens(data)
                tokens = util.normalize(tokens)
                if len(tokens) > self.min_word and len(tokens) < self.max_word:
                    fredict = nltk.FreqDist(tokens)
                    for k,v in fredict.items():
                        p = Pair(self.fileId,v/len(tokens))
                        if k not in index_table:
                            index_table[k] = [p]
                        else:
                            index_table[k].append(p)
                    self.indexedUrl.append(url)
                    self.fileId += 1
                else:
                    self.logfile.write(f"{url} {len(tokens)}\n")
                file.close()
            except Exception as e:
                self.logfile.write(f"{d} {e}\n")
                
        return index_table
    
    
    def batch_of_documents(self,documents,size):
        """
        generator function
        """
        num_of_files = len(documents)
        for i in range(0,num_of_files,size):
            yield documents[i:min(i+size,num_of_files)]

    def index_all_Doc(self):
        self.logfile = open(util.LOG_FILE,"w")
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
        print(f"number of pages indexed {len(self.indexedUrl)}")
        util.store_data(self.indexedUrl,util.PRE_INDEXED_URL_PATH )
        util.store_data(self.unique_tokens,util.PRE_TOKEN_DATA_PATH )
        self.logfile.close()


