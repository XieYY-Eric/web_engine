from lib2to3.pgen2.tokenize import tokenize
import pickle
from collections import namedtuple

Pair = namedtuple("Pair", ["first", "second"])
class indexer:
    #IMPORTANT: CALL INDEXER CLOSE BEFORE TERMINATING THE PROGRAM!!!
    #TO-DO M2: add tf-idf score. Run unique tokens through and take the last in the array and the second of the pair and divide by word count
    def __init__(self, restart = False):
        '''make the indexer add the inverted index filepaths and the indexer of that'''
        if restart:
            self.indices = {} #its token: array that holds pairs, the pair is the docID and the word freq
        else:
            with open("./data/inverted_index_a","rb") as f:
                self.indices = pickle.load(f)

    def add_data(self, tokens, fileID):
        '''adds data to the indices'''
        organized = self.organize_data(tokens)
        for token in organized.keys :
            self.indices[token].append(Pair(fileID, organized[token]))
                                       

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


