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



def getTokens(content):
    soup = BeautifulSoup(content)
    data = ""
    for section in soup.find_all('p'):
        text = section.get_text()
        data += (text + " ")
    tokens = nltk.tokenize.word_tokenize(data)
    return tokens

def is_html(content):
    html_tag = [r"<!DOCTYPE html",r"<!DOCTYPE HTML"]
    for tag in html_tag:
        if tag in content:
            return True
    return False #expecting the top few string are html_tag


def normalize(tokenlist):
    #remove punctuation
    filter_tokens = [re.sub(r'[^\w\s]', '', token) for token in tokenlist]
    #lower
    filter_tokens = [token.lower() for token in tokenlist]
    #stemming
    #filter_tokens = [PS.stem(token,to_lowercase=True) for token in tokenlist]
    return filter_tokens

def read_data(filename):
    with open(filename,"rb") as f:
        return pickle.load(f)


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
                    tokens = normalize(tokens)
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
                file.close()
            except Exception as e:
                print(f"filename {d} Exception {e}")
        return index_table
    
    def batch_of_documents(self,documents,size):
        num_of_files = len(documents)
        for i in range(0,num_of_files,size):
            yield documents[i:min(i+size,num_of_files)]

    def index_all_Doc(self):
        begin = time.time()
        end = time.time()
        number_of_batch = math.ceil(len(self.allDocuments)/self._batch_size)
        for i,batch in enumerate(self.batch_of_documents(self.allDocuments,self._batch_size)):
            file_to_store = self._partial_index_file_prefix+str(i)+".p"
            index_table = self.index_DocBatch(batch)
            f = open(file_to_store,"wb") #write new file
            pickle.dump(index_table,f)
            f.close
            end = time.time()
            print(f"Batch {i+1}/{number_of_batch} completed {(end-begin):.3f}s")
            begin = end
        print(f"number of pages indexed {len(self.indexedDocument)}")
        self.sort_to_partial_file(number_of_batch)
        self.merge_files(number_of_batch)
    

    def sort_to_partial_file(self, number_of_batch):
        for i in range(0, number_of_batch):
            file_to_open = self._partial_index_file_prefix+str(i)+".p"
            file_to_write = self._partial_index_file_prefix+str(i)+".txt"
            data = read_data(file_to_open)
            partial_index = open(file_to_write, "a")
            for j in sorted(data.keys()):
                info = data[j]
                info.sort(key=lambda x:x.DocID)
                partial_index.write(f"{j}: {info}/n")
            partial_index.close()

    def merge_files(self, number_of_files):
        txtfiles = []
        filelines = []
        tokens = []
        empty = 0
        for i in range(0,number_of_files):
            f = self._partial_index_file_prefix+str(i)+".txt"
            txtfiles.append(open(f, "r"))
            
            x = txtfiles[i].readline() #first line is read
            if x == "": #file empty
                filelines.append("")
                tokens.append("")
                empty += 1
            else:
                filelines.append(self.process_index_txt(x)) #a dic with just that token  
                tokens.append(filelines[i].keys()[0])

        index = open("main_index.txt", "a")

        while empty < number_of_files:
             sorted_tokens = tokens
             sorted_tokens.sort() 
             to_write = {}
             lines_to_replace = []

             y = 0
             while sorted_tokens[x] == "": #ignore any empty files
                 y += 1
             
             for l in range(0, len(tokens)):
                 if sorted_tokens[y] == tokens[l]:
                     lines_to_replace.append(l)
                     if tokens[l].keys()[0] in to_write.keys(): #just add it on
                         to_write[tokens[l].keys()[0]].extend(tokens[l])
                     else:
                         to_write[tokens[l].keys()[0]] = tokens[l] 
             #have the line to write now
             index.write(f"{sorted_tokens[y]}: {to_write[sorted_tokens[x]]}/n")

             for each in lines_to_replace:
                 nline = txtfiles[each].readline
                 if nline == "":
                    filelines.append("")
                    tokens.append("")
                    empty += 1
                 else:
                    filelines[each] = self.process_index_txt(x) #a dic with just that token  
                    tokens[each] = filelines[i].keys()[0]

        for j in txtfiles: #close all the files
            j.close()
        index.close

        #END OF INDEX MAKING

    def process_index_txt(self,line):
        '''token: [list of pairs], returns a dict'''
        data = {}
        token = line[:line.index(":")]
        x = line[line.index("[")+5:line.index("]")]
        info = line.split("Pair")

        for each in range(0, len(info)):
            pairtuple =info[each][1:info[each].index(")")]
            if each == 0:
                data[token] = [Pair(int(pairtuple[0][pairtuple[0].index("=")+1:]),int(pairtuple[1][pairtuple[1].index("=")+1:]))]
            else:
                data[token].append([Pair(int(pairtuple[0][pairtuple[0].index("=")+1:]),int(pairtuple[1][pairtuple[1].index("=")+1:]))])
        return data


    '''
    def merge(self,file1,file2):
        with open(file1,"rb") as f:
            table1 =  pickle.load(f)
        with open(file2,"rb") as f:
            table2 = pickle.load(f)
        for token in table2:
            if token in table1.keys():
                for k, v in table1[token] + table2[token]:
                    table1[k] = table1.get(k, 0) + v
        with open(fileName, 'wb') as f:
            pickle.dump(table1, f)    
        '''
        
            


              


