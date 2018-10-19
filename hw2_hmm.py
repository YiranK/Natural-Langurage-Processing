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
from math import log,inf

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

        # for later Laplace smoothing
        # self.tag_num = 0
        # self.tag_list = []
        # for t in self.tags.keys():
        #     if self.tags[t] > 0:
                # self.tag_num += 1
                # self.tag_list.append(t)
        # self.tag_np = np.array(self.tag_list)
        # deal with the word counts <= 5
        self.words['UNK'] = 0.0
        unk_word = []
        for word, count in self.words.items():
            if count <= 5:
                self.words[word] = 0 # or delete?
                self.words['UNK'] += 1
                unk_word.append(word) # or to set unique?
        print(unk_word)
        # todo2: count ti + wi, ti-1 + ti
        # self.tag_types = []
        # self.word_types = []
        for sen in data:
            for i in range(len(sen)-1):
                word = sen[i].word
                # if word in unk_word:
                if self.words[word] <= 5:
                    word = 'UNK'
                # if word not in self.word_types:
                #     self.word_types.append(word)
                # if sen[i].tag not in self.tag_types:
                #     self.tag_types.append(sen[i].tag)
                #print(sen[i].tag, sen[i].word)
                self.t2w[sen[i].tag][word] += 1
                self.t2t[sen[i].tag][sen[i+1].tag] += 1
            
            word = sen[-1].word
            if self.words[word] <= 5:
            # if word in unk_word:
                word = 'UNK'
            # if word not in self.word_types:
            #     self.word_types.append(word)
            # if sen[-1].tag not in self.tag_types:
            #     self.tag_types.append(sen[-1].tag)

            self.t2w[sen[-1].tag][word] += 1

        # count tag types, word types 
        # todo: construct an index
        # self.tag_types = []
        # for tag in self.tags.keys():
        #     if self.tags[tag] > 0:
        #         self.tag_types.append(tag)
        # self.word_types = []
        # for word in self.words.keys():
        #     if self.words[word] > 0:
        #         self.word_types.append(word)
        # self.tag_num = len(self.tag_types)
        # self.word_num = len(self.word_types)
        # print(self.tag_num, self.word_num)

        # self.trellis = np.zeros((tag_num, word_num), dtype=np.float) # trellis, (tags x words)

        # previous tag count
        self.tag_tran_count = defaultdict(float)
        for tag1 in self.tags.keys():
            tag1_count = 0
            for tag2 in self.t2t[tag1].keys():
                if self.t2t[tag1][tag2] > 0:
                    tag1_count += 1
            self.tag_tran_count[tag1] = tag1_count

        # tag total
        self.tag_total = 0
        for tag in self.tags.keys():
            self.tag_total += self.tags[tag]


    def getProb_t2w(self, tag, word):
        # print(tag, word, self.t2w[tag][word], self.tags[tag])
        t2w_prob = self.t2w[tag][word]/self.tags[tag]# log?
        if t2w_prob != 0:
            return log(t2w_prob)
        else:
            return -inf

    def getProb_t2t(self, tag1, tag2):
        # print(tag1)
        t2t_prob = (self.t2t[tag1][tag2]+1)/(self.tags[tag1]+self.tag_tran_count[tag1]) #transition probability. the condition is the proceeding tag is tag1
        if t2t_prob != 0:
            return log(t2t_prob)
        else:
            return -inf

    def getProb_t(self, tag):
        t_prob = self.tags[tag] / self.tag_total
        if t_prob != 0:
            return log(t_prob)
        else:
            return -inf

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
            print(len(sen),len(vitTags))
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
        filtered_freq_words = [x if self.words[x] > 5 else 'UNK' for x in words]
        words = filtered_freq_words
        for w in words:
            print(w)

        tag_list = list(self.tags.keys())
        trellis = []
        backpointer = []
        for i in range(len(tag_list)):
            trellis.append([-inf] * len(words))
            backpointer.append([0] * len(words))

        for i in range(len(tag_list)):
            if self.t2w[tag_list[i]][words[0]] > 0:
                print("Initialize, tag {} : {} for the first word {}, prob_t:{}, getProb_t2w:{}".\
                    format(i,tag_list[i],words[0],self.getProb_t(tag_list[i]),self.getProb_t2w(tag_list[i], words[0])))
                trellis[i][0] = self.getProb_t(tag_list[i]) + self.getProb_t2w(tag_list[i], words[0])

        for j in range(1, len(words)):
            for i in range(len(tag_list)):
                if self.t2w[tag_list[i]][words[j]] > 0:
                    emission_prob = self.getProb_t2w(tag_list[i], words[j])

                    max_tran_prob = -inf
                    max_tran_t = 0
                    for k in range(len(tag_list)):
                        tran_prob = trellis[k][j-1] + self.getProb_t2t(tag_list[k],tag_list[i])
                        if tran_prob > max_tran_prob:
                            max_tran_prob = tran_prob
                            max_tran_t = k

                    trellis[i][j] = max_tran_prob + emission_prob
                    backpointer[i][j] = max_tran_t
                    print("for now word {} with tag {}, the best tran_prob is {} with prev tag {}.".format(j,i,max_tran_prob,max_tran_t))

        max_last_prob = -inf
        max_last_t = 0
        for i in range(len(tag_list)):
            if trellis[i][len(words)-1] > max_last_prob:
                max_last_prob = trellis[i][len(words)-1]
                max_last_t = i

        t = max_last_t
        tag_seq = [tag_list[t]]
        for j in range(len(words)-1, 0, -1):
            t = backpointer[t][j]
            tag_seq.append(tag_list[t])
        
        # for i, w in enumerate(words):
        #     if self.words[w] <= 5:
        #         w = "UNK"
        #     for t in self.tags.keys():
        #         # emission
        #         prob = self.getProb_t2w(t, w)
        #         # print(t,w,prob)
        #         if prob == 0:
        #             continue
        #         else:
        #             prob = log(prob)

        #         # transition
        #         # todo : how to pruning
        #         if i != 0:
        #             tran_max = -1000
        #             t_pre_best = ""
        #             for t_pre in self.tags.keys():
        #                 if words[i-1] not in self.trellis[t_pre].keys():
        #                     continue

        #                 # print(self.getProb_t2t(t_pre, t))
        #                 tran = log(self.getProb_t2t(t_pre, t)) + self.trellis[t_pre][words[i-1]].prob
        #                 print("tran", tran)
        #                 if tran > tran_max:
        #                     tran_max = tran
        #                     t_pre_best = t_pre

        #             prob += tran_max
        #             self.trellis[t][w] = cell(prob, t_pre_best)
        #         else:
        #             self.trellis[t][w] = cell(prob, "init") #does it have some start symbol?
        #         print(t, w, self.trellis[t][w].prob, self.trellis[t][w].back)

        # tag_seq = []
        # max_last_col = -10
        # max_last_tag = ""
        # for t in self.tags.keys():
        #     # print(t, max_last_col, max_last_tag)
        #     if words[-1] in self.trellis[t].keys():
        #         print(self.trellis[t][words[-1]].prob)
        #     if words[-1] in self.trellis[t].keys() and self.trellis[t][words[-1]].prob > max_last_col:
        #         max_last_col = self.trellis[t][words[-1]].prob
        #         max_last_tag = t
        # # t = max(self.trellis, key=lambda k:self.trellis[k][words[-1]]) # find the max value in last column, returns the key tag
        # tag_seq.append(max_last_tag)
        # t = max_last_tag
        # for i in range(len(words)-1, 0, -1):
        #     print(i, t)
        #     t = self.trellis[t][words[i]].back
        #     print(t)
        #     tag_seq.append(t)
        # tag_seq.reverse()
        print("tag_seq", tag_seq)
        # returns the list of Viterbi POS tags (strings)
        # return ["NULL"]*len(words) # this returns a dummy list of "NULL", equal in length to words
        return tag_seq

if __name__ == "__main__":
    tagger = HMM()
    tagger.train('train.txt')
    print("train finished")
    tagger.test('test.txt', 'out.txt')
