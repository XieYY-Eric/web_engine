from lib2to3.pgen2.tokenize import tokenize
import pickle
from collections import namedtuple
import json
from bs4 import BeautifulSoup
import re
from nltk.stem import PorterStemmer
import nltk
import time

Pair = namedtuple("Pair", ["first", "second"])
class indexer:
    #IMPORTANT: CALL INDEXER CLOSE BEFORE TERMINATING THE PROGRAM!!!
    #TO-DO M2: add tf-idf score. Run unique tokens through and take the last in the array and the second of the pair and divide by word count
    def __init__(self, files,tokens,index_file_prefix="./data/Index_tables/"):
        '''make the indexer add the inverted index filepaths and the indexer of that'''
        self._files = files
        self._tokens = tokens
        self._index_table = {}
        self._indexed_file_count = 0
        self._partial_index_file_prefix = index_file_prefix
        self._batch_size = 512
        self.reset()

    def reset(self):
        for token in self._tokens:
            self._index_table[token] = []


    def index_DocBatch(self,documents_batch):
        for d in documents_batch:
            file = open(d,"r")
            data = json.load(file)
            data = data["content"]
            data = BeautifulSoup(data).get_text()
            tokens = nltk.tokenize.word_tokenize(data)
            tokens = self.normalize(tokens)
            self.add_data(tokens,self._indexed_file_count)
            self.reset()
            self._indexed_file_count += 1
            file.close()
    
    def batch_of_documents(self,documents,size):
        num_of_files = len(documents)
        for i in range(0,num_of_files,size):
            yield documents[i:min(i+size,num_of_files)]

    def index_all_Doc(self):
        batches = self.batch_of_documents(self._files,self._batch_size)
        begin = time.time()
        end = time.time()
        for i,batch in enumerate(batches):
            file_to_store = self._partial_index_file_prefix+str(i)+".p"
            self.index_DocBatch(batch)
            f = open(file_to_store,"x") #create new file
            f.close()   #close new file
            f = open(file_to_store,"wb") #write new file
            pickle.dump(self._index_table)
            f.close
            self.reset()
            end = time.time()
            print(f"Batch {i}/{len(batches)} completed {(end-begin):.3f}s")
            begin = end

        

    def add_data(self, tokens, fileID):
        '''adds data to the indices'''
        organized = self.organize_data(tokens)
        for token,value in organized.items():
            self._index_table[token].append(Pair(fileID, value))


    def normalize(self,tokenlist):
        PS = PorterStemmer()
        #remove punctuation
        filter_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokenlist]
        #lower
        filter_tokens = [token.lower() for token in tokenlist]
        #stemming
        filter_tokens = [PS.stem(token,to_lowercase=True) for token in tokenlist]
        return filter_tokens
                                       

    def organize_data(self, tokens):
        '''returns a dictionary of tokens and their frequency'''
        organizer = {}
        for token in tokens:
            if token in organizer.keys():
                organizer[token] += 1
            else:
                organizer[token] = 1
        return organizer

    def close(self):
        '''Stores data back into the pickle file'''
        with open("./data/inverted_index_a","wb") as f:
            pickle.dump(self.indices,f)


