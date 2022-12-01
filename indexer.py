from collections import namedtuple
import json
from turtle import towards
from bs4 import BeautifulSoup
import nltk
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
    important_tokens = set()

    if is_html(content):
        #is html file, index it
        soup = BeautifulSoup(content)
        #### all find paragraghs
        # for section in soup.find_all('p'):
        #     text = section.get_text()
        #     data += (text + " ")
        #### all find text
        data = soup.get_text(separator=" ",strip=True)
        regex_expression = r"[a-zA-Z\d]+"
        tokens = re.findall(regex_expression, data)

        # get all important tokens from doc
        for tags in soup.find_all(selector, limit=5):
            text = tags.get_text(separator=" ", strip=True)
            tag_tokens = re.findall(regex_expression, text)
            #print(set(util.normalize(list(tag_tokens))))
            important_tokens.update(set(util.normalize(list(tag_tokens))))
    else:
        #non html, treat it as raw page
        data = content
        regex_expression = r"[a-zA-Z\d]+"
        tokens = re.findall(regex_expression,data)

    return tokens, important_tokens


def selector(tag):
    return tag.name == 'title' or tag.name == 'bold' or tag.name == 'h1' or tag.name == 'h2' or tag.name == 'h3'

#using logarithmically scaled frequency: tf(t,d) = log (1 + ft,d)


Pair = namedtuple("Pair", ["DocID", "count", "importance"])
class indexer:
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
        self.number_of_partial_table = 0
        self.logfile = None

    def index_DocBatch(self, documents_batch):
        index_table = {}
        for d in documents_batch:
            try:
                file = open(d, "r")
                data = json.load(file)
                url = data["url"]
                data = data["content"]
                ##process the string, get the token from string
                tokens, important_tokens = getTokens(data)
                tokens = util.normalize(tokens)
                if len(tokens) > self.min_word and len(tokens) < self.max_word:
                    fredict = nltk.FreqDist(tokens)
                    for k, v in fredict.items():
                        importance = 1 if k in important_tokens else 0
                        p = Pair(self.fileId, v, importance)
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
            print(f"Batch {i + 1}/{number_of_batch} started")
            file_to_store = self._partial_index_file_prefix+str(i)+".p"
            index_table = self.index_DocBatch(batch)
            self.unique_tokens = self.unique_tokens.union(set(index_table.keys()))
            util.store_data(index_table,file_to_store)
            self.number_of_partial_table = i
            end = time.time()
            print(f"Batch {i+1}/{number_of_batch} completed {(end-begin):.3f}s")
            begin = end
        print(f"number of pages indexed {len(self.indexedUrl)}")
        util.store_data(self.indexedUrl,util.PRE_INDEXED_URL_PATH )
        util.store_data(self.unique_tokens,util.PRE_TOKEN_DATA_PATH )
        self.logfile.close()
        print("Start formating partial table...")
        tokens = list(self.unique_tokens)
        self.format_all_files(self.number_of_partial_table,tokens)
        self.merge_all_files(self.number_of_partial_table,len(self.indexedUrl))

    def format_file(self,tokens,partial_tables,filename):
        #get all the tokens
        f = open(filename,"w",encoding="utf-8")
        for token in tokens:
            if token in partial_tables:
                data =[(tuple[0],tuple[1],tuple[2]) for tuple in partial_tables[token]]
                f.write(f"{token}:{data}\n")
            else:
                f.write(f"{token}:[]\n")
        f.close()

    def format_all_files(self,number_of_files,tokens):
        #formet the file, make sure every token appear in the
        tokens.sort()
        for i in range(number_of_files):
            partial_table = util.read_data(util.INDEX_TABLE_PREFIX+str(i)+".p")
            self.format_file(tokens,partial_table,"./data/Index_tables/"+str(i)+".txt")
            print(f"file {i} formatted completed")

    def merge_all_files(self,number_of_files,total_url_indexed):
        filenames = [util.INDEX_TABLE_PREFIX+str(i)+".txt" for i in range(number_of_files)]
        files = [open(filename,"r") for filename in filenames]
        next_lines = [files[i].readline() for i in range(number_of_files)]
        line = next_lines[0]
        file_to_write = util.INDEX_TABLE_NAME
        f = open(file_to_write,"w",encoding="utf-8")
        print_every = 10000
        count = 0
        while line:
            values = []
            token = ""
            for this_line in next_lines:
                token,value = this_line.strip().split(":")
                values.extend(eval(value))
            #convert count to idf-tf
            processed_value = []
            d = len(values)
            for (DID,count,importance) in values:
                df = math.log(1+count)
                idf = max(0,math.log(total_url_indexed/(1+d)))
                processed_value.append((DID,df*idf,importance))
            f.write(f"{token}:{processed_value}\n")
            count += 1
            if count%print_every == 0:
                print(f"merging token {count}")
            next_lines = [files[i].readline() for i in range(number_of_files)]
            line = next_lines[0]
        f.close()
        closing = [file.close() for file in files]


def main():
    #read file names from DEV folder
    dataset = util.get_all_file_names(util.DATA_PATH)
    #indexing pages
    myindexer = indexer(dataset,2048,util.INDEX_TABLE_PREFIX,util.MIN_WORD,util.MAX_WORD)
    myindexer.index_all_Doc() #COMMENT out this if you dont wanna computing all over again
    all_tokens = list(util.read_data(util.PRE_TOKEN_DATA_PATH))
    all_urls_size = len(util.read_data(util.PRE_INDEXED_URL_PATH))
    number_of_partial_table = 10 # replace this value with the actual number
    myindexer.format_all_files(number_of_partial_table,all_tokens)
    myindexer.merge_all_files(number_of_partial_table,all_urls_size)
    




if __name__ == "__main__":
    main()
