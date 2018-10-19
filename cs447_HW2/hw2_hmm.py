########################################
## CS447 Natural Language Processing  ##
##           Homework 2               ##
##       Julia Hockenmaier            ##
##       juliahmr@illnois.edu         ##
########################################
##
## Part 1:
## Train a bigram HMM for POS tagging
##
import os.path
import sys
from operator import itemgetter
from collections import defaultdict
from math import log

# Unknown word token
UNK = 'UNK'

# Class that stores a word and tag together
class TaggedWord:
    def __init__(self, taggedString):
        parts = taggedString.split('_');
        self.word = parts[0]
        self.tag = parts[1]

# Class for cell in Verbiti
class cell:
	def __init__(self, prob, back):
		self.prob = prob
		self.back = back

# Class definition for a bigram HMM
class HMM:
### Helper file I/O methods ###
    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads a labeled data inputFile, and returns a nested list of sentences, where each sentence is a list of TaggedWord objects
    def readLabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                raw = line.split()
                sentence = []
                for token in raw:
                    sentence.append(TaggedWord(token))
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s does not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script

    ################################
    #intput:                       #
    #    inputFile: string         #
    #output: list                  #
    ################################
    # Reads an unlabeled data inputFile, and returns a nested list of sentences, where each sentence is a list of strings
    def readUnlabeledData(self, inputFile):
        if os.path.isfile(inputFile):
            file = open(inputFile, "r") # open the input file in read-only mode
            sens = [];
            for line in file:
                sentence = line.split() # split the line into a list of words
                sens.append(sentence) # append this list as an element to the list of sentences
            return sens
        else:
            print("Error: unlabeled data file %s ddoes not exist" % inputFile)  # We should really be throwing an exception here, but for simplicity's sake, this will suffice.
            sys.exit() # exit the script
### End file I/O methods ###

    ################################
    #intput:                       #
    #    unknownWordThreshold: int #
    #output: None                  #
    ################################
    # Constructor
    def __init__(self, unknownWordThreshold=5):
        # Unknown word threshold, default value is 5 (words occuring fewer than 5 times should be treated as UNK)
        self.minFreq = unknownWordThreshold
        ### Initialize the rest of your data structures here ###
        self.tags = defaultdict(float)
        self.words = defaultdict(float)
        self.t2w = defaultdict(lambda: defaultdict(float)) # ti -> wi
        self.t2t = defaultdict(lambda: defaultdict(float)) # ti-1 -> ti
        self.trellis = defaultdict(lambda: defaultdict(cell))


    ################################
    #intput:                       #
    #    trainFile: string         #
    #output: None                  #
    ################################
    # Given labeled corpus in trainFile, build the HMM distributions from the observed counts
    def train(self, trainFile):
        data = self.readLabeledData(trainFile) # data is a nested list of TaggedWords
        print("Your first task is to train a bigram HMM tagger from an input file of POS-tagged text")

        # todo1: count the single word, deal with UNK; count single tag
        for sen in data:
        	for i in range(len(sen)):
        		self.tags[sen[i].tag] += 1
        		self.words[sen[i].word] += 1
        		
        # deal with the word counts <= 5
        unk_word = []
        for word, count in self.words.items():
        	if count <= 5:
        		self.words[word] = 0 # or delete?
        		self.words['UNK'] += 1
        		unk_word.append(word) # or to set unique?

        # todo2: count ti + wi, ti-1 + ti
        for sen in data:
        	for i in range(len(sen)-1):
        		if sen[i].word in unk_word:
        			sen[i].word = 'UNK'
        		self.t2w[sen[i].tag][sen[i].word] += 1
        		self.t2t[sen[i].tag][sen[i+1].tag] += 1
        		
        	if sen[-1].word in unk_word:
        		sen[-1].word = 'UNK'
        	self.t2w[sen[-1].tag][sen[-1].word] += 1

        # count tag types, word types 
        # todo: construct an index
        self.tag_types = []
        for tag in self.tags.keys():
        	if self.tags[tag] > 0:
        		self.tag_types.append(tag)
        self.word_types = []
        for word in self.words.keys():
        	if self.words[word] > 0:
        		self.word_types.append(word)
        self.tag_num = len(self.tag_types)
        self.word_num = len(self.word_types)

        # self.trellis = np.zeros((tag_num, word_num), dtype=np.float) # trellis, (tags x words)


    def getProb_t2w(tag, word):
    	return self.t2w[tag][word]/self.tags[tag]# log?

    def getProb_t2t(tag1, tag2):
    	return self.t2t[tag1][tag2]/self.tags[tag1] #transition probability. the condition is the proceeding tag is tag1


    ################################
    #intput:                       #
    #     testFile: string         #
    #    outFile: string           #
    #output: None                  #
    ################################
    # Given an unlabeled corpus in testFile, output the Viterbi tag sequences as a labeled corpus in outFile
    def test(self, testFile, outFile):
        data = self.readUnlabeledData(testFile)
        f=open(outFile, 'w+')
        for sen in data:
            vitTags = self.viterbi(sen)
            senString = ''
            for i in range(len(sen)):
                senString += sen[i]+"_"+vitTags[i]+" "
            print(senString)
            print(senString.rstrip(), end="\n", file=f)

    ################################
    #intput:                       #
    #    words: list               #
    #output: list                  #
    ################################
    # Given a list of words, runs the Viterbi algorithm and returns a list containing the sequence of tags
    # that generates the word sequence with highest probability, according to this HMM
    def viterbi(self, words):
        print("Your second task is to implement the Viterbi algorithm for the HMM tagger")
        
        for w, i in enumerate(words):
        	for t in tag.types:
        		# emission
        		prob = self.getProb_t2w(t, w)
        		if prob == 0:
        			continue
        		else:
        			prob = math.log(prob)

        		# transition
        		# todo : how to pruning
        		if i != 0:
        			tran_max = 0.0
        			t_pre_best = ""
        			for t_pre in tag.types:
        				if self.trellis[t_pre][words[i-1]].prob == 0:
        					continue
        				tran = math.log(self.getProb_t2t(t_pre, t)) + math.log(self.trellis[t_pre][words[i-1]].prob)
        				if tran > tran_max:
        					tran_max = tran
        					t_pre_best = t_pre

        			prob *= tran_max
        			trellis[t][w] = cell(prob, t_pre_best)
        		else:
        			trellis[t][w] = cell(prob, "init") #does it have some start symbol?

        tag_seq = []
        t = max(trellis, key=lambda k:trellis[k][words[-1]]) # find the max value in last column, returns the key tag
        tag_seq.append(t)
        for i in range(-1, 0, -1):
        	t = trellis[t][words[i]].back
        	tag_seq.append(t)
        tag_seq.reverse()
        # returns the list of Viterbi POS tags (strings)
        # return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words
        return tag_seq

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    tagger.test('test.txt', 'out.txt')
