"""
Filename: word2vec.py

Despcription:
This file performs word2vec conversion using a given vocabulary file and a corpus
file. The converted vectors will be saved to vectors.txt file. This file uses the
skip-gram method along with negative sampling algorithm. 
Code by Jashanvir Singh Taggar and Anjandeep Singh Sahni
"""

import numpy as np
from collections import deque
import math
import time
import random
from random import randint
import sys
import traceback
from tqdm import tqdm

#Global parameters.
gVocab = set()      #Set of Vocabulary. Duplicates are removed by 'set' operations.
gVocabDict = {}     #Dict of vocabulary. Will have index (value) for each word (key).
gVocabSize = 0      #Vocabulary size.
gTrainData = []     #List to maintain all center words.
gLabelData = []     #List to maintain all context words for center words.
gCrpsWrdFreq = {}   #Dict to maintain frequency (value) of each word (key) in corpus.
gWindowSize = 3     #Center word neighbourhood window size.
gTotalWndwSize = ((2 * gWindowSize) + 1)   #Total window size including center word.
gHidLyrSize = 200    #Size of hidden layer in neural network.
gAlpha = 0.01       #Learning rate.
gTotalEpochs = 5    #Number of learning epochs.
gNegSamplePow = 0.75    #Power for word frequency. Used in negative sampling.
gNegSampleArr = []      #List of words' indices for negative sampling.
gNegSampleArrMaxSize = 1000000  #Max size of word array for negative sampling.
gNumNegSamples = 4  #Number of context words to choose for negative sampling.
#gSubSamplingRate controls how much subsampling occurs.
#Smaller value means words are less likely to be kept.
gSubSamplingRate = 0.001

#Sigmoid function.
def sigmoid(z):
    return 1 / (1 + math.exp(-z))

#This function reads the vocabulary file
#and adds words to gVocab set.
#It also creates a vocab dict with (word:index) pairs.
def readVocab(filename):
    try:
        global gVocabSize
        global gVocab
        global gVocabDict
        with open(filename,'r') as f:
            for line in f:
                for word in line.split():
                    gVocab.add(word)
                    gVocabDict.update({word:gVocabSize})
                    gVocabSize += 1
        return 0
    except: #Catch all exceptions.
        e = sys.exc_info()[0]
        print("Error: %s" %e)
        traceback.print_exc()
        return -1

#This function reads the corpus file
#and performs the following tasks:
#1. gTrainData = target/center words.
#2. gLabelData = context words.
#3. gCrpsWrdFreq = frequency of each center word.
def readcorpus(filename):
    global gVocab
    global gTrainData
    global gLabelData
    global gCrpsWrdFreq
    global gWindowSize
    global gTotalWndwSize
    global gSubSamplingRate
    crpsWrdCnt = 0
    queue = deque()     #Queue to maintain sliding window around center word.
    #First generate frequencies for each word in corpus and vocab.
    try:
        with open(filename,'r') as f:
            for line in f:
                for word in line.split():
                    crpsWrdCnt += 1
                    #Check if word is in gVocab.
                    if word in gVocab:
                        #Check if word is already added
                        #to gCrpsWrdFreq dict.
                        if word not in gCrpsWrdFreq:
                            gCrpsWrdFreq[word] = 1
                        else:
                            gCrpsWrdFreq[word] +=1
    except: #Catch all exceptions.
        e = sys.exc_info()[0]
        print("Error: %s" %e)
        traceback.print_exc()
        return -1
    #Now, generate training set.
    try:
        with open(filename,'r') as f:
            for line in f:
                for word in line.split():
                    queue.append(word)  #Add new word to queue.
                    #We skip first few words till we reach the total window size.
                    if len(queue) < gTotalWndwSize:
                        continue
                    #If queue has increased beyond total window size,
                    #pop the leftmost word to shift the window.
                    if len(queue) > gTotalWndwSize:
                        queue.popleft()
                    #Here we will always have exactly
                    #gTotalWndwSize words in queue.
                    #Check if center word is in gVocab.
                    if queue[gWindowSize] in gVocab:
                        #Subsampling.
                        #Calculate probability of keeping the word.
                        ssProb = float((math.sqrt(float(gCrpsWrdFreq[queue[gWindowSize]] /
                                                (gSubSamplingRate * crpsWrdCnt))) + 1) *
                                                (float((gSubSamplingRate * crpsWrdCnt) /
                                                gCrpsWrdFreq[queue[gWindowSize]])))
                        #Change seed for random number generator.
                        random.seed(randint(0, 1000) * time.time())
                        #Generate random number between 0 and 1.
                        ssRandom = float(random.randint(1,65536) / 65535)
                        if ssProb < ssRandom:
                            continue    #Skip due to subsampling.
                        #Loop through all context words.
                        for i in range(0, gTotalWndwSize):
                            #If current word is center word, skip.
                            if i == gWindowSize:
                                continue
                            #Check if context word is in gVocab.
                            if queue[i] in gVocab:
                                #Add center word to gTrainData.
                                gTrainData.append(queue[gWindowSize])
                                #Add context word to gLabelData.
                                gLabelData.append(queue[i])
        return 0
    except: #Catch all exceptions.
        e = sys.exc_info()[0]
        print("Error: %s" %e)
        traceback.print_exc()
        return -1

#This function generates the word array used for
#negative sampling. Words with higher probability
#are present more times in the gNegSampleArr array.
#Note: gNegSampleArr only has indices of words from gVocabDict.
def initUnigramTable():
    global gNegSampleArr
    global gVocabDict
    global gCrpsWrdFreq
    global gNegSamplePow
    global gNegSampleArrMaxSize
    #Used gVocabDict to generate gNegSampleArr.
    probArr = []        #Probability of each word.
    totalProb = 0.0     #Sum of probability of each word.
    arrWordCount = 0    #Number of times a word appears in array.
    #P(word) = (freq(word)^3/4)/sumOfAll(freq(word))
    for index, (word, freq) in enumerate(gCrpsWrdFreq.items()):
        probArr.append(pow(freq, gNegSamplePow))
        totalProb = totalProb + probArr[index]
    for index, (word, freq) in enumerate(gCrpsWrdFreq.items()):
        probArr[index] = probArr[index]/totalProb
        arrWordCount = int(probArr[index] * gNegSampleArrMaxSize)
        #Add index of word in array arrWordCount times.
        gNegSampleArr = gNegSampleArr + ([gVocabDict[word]] * arrWordCount)

#This function performs negative sampling.
#It returns a list of randomly chosen negative samples.
def negativeSampling():
    global gNumNegSamples
    negSamples = []    #List of randomly chosen context word indices.
    arrLen = len(gNegSampleArr)
    #Change seed for random number generator.
    random.seed(randint(0, 1000) * time.time())
    for i in range(gNumNegSamples):
        #Generate random integer between 0 and array length.
        randomIndex = randint(0, arrLen - 1)
        negSamples.append(gNegSampleArr[randomIndex])
    return negSamples

#This function performs forward propogation on training data.
#Intution: This calculates the probability of each word
#to appear in the context of the center word.
def forwardPropogation(targetIndex, w1, w2):
    #Pick the weights for target/center word.
    #This is also equal to the hidden layer node values.
    centerWord = w1[targetIndex,:]
    #Multiply target/center word with each context word.
    #This is equal to the final output of the neural network.
    probContextWords = np.matmul(centerWord,w2)
    return centerWord, probContextWords

#This function performs backwards propogation using negative sampling.
#It will calculate the new values of weights W1 and W2 for 1 context word
#and gNumNegSamples negative sample words.
def backPropogation(targetIndex, contextIndex, nsIndices, w1, w2, hiddenLayer):
    global gAlpha
    #Calculate EI for context word. EI = sig(inputWord * contextWord) - 1
    EIContext = sigmoid(np.matmul(w1[targetIndex,:],w2[:,contextIndex])) - 1
    #Calculate EH for context word. EH = EI * contextWord
    EHContext = EIContext * w2[:,contextIndex]
    #Update the W2 weight for context word.
    #contextWord = contextWord - (learningRate * EI * hiddenLayer)
    w2[:,contextIndex] -=  (gAlpha * EIContext * hiddenLayer)

    #Calculate EI and EH for all neg samples.
    EHNeg = np.transpose(np.zeros(len(w2[:,1])))  #Null vector.
    EINeg = [0]
    count = 1
    for vec in nsIndices:
        EHNeg += sigmoid(np.matmul(w1[targetIndex,:], w2[:,vec])) * w2[:,vec]
        EINeg = np.hstack((EINeg, sigmoid(np.matmul(w1[targetIndex,:], w2[:,vec]))))
        #EINeg is a horizontal vector. Each entry has EI for a neg sample.
        #Update the W2 weight for neg sample.
        w2[:,vec] -=  gAlpha * EINeg[count] * hiddenLayer
        count += 1

    #Updating the W1 weight for target word.
    EHSum = EHContext + EHNeg
    w1[targetIndex,:] -=  gAlpha * EHSum
    return w1, w2

#Main. Entry point.
if __name__ == "__main__" :
    #Print some introductory information.
    print("=="*20)
    print("Null Pointer Tech: Word2Vec")
    print("=="*20)
    print("--"*20)
    print("Hyperparameters:")
    print("--"*20)
    print('Learning Rate: {}'.format(gAlpha))
    print('Number of Negative Samples: {}'.format(gNumNegSamples))
    print('Hidden Layer Size: {}'.format(gHidLyrSize))
    print('Window Size: {}'.format(gWindowSize))
    print('Subsampling Rate: {}'.format(gSubSamplingRate))
    print("--"*20)

    #Read and populate vocabulary.
    print("Reading vocabulary file \"vocab.txt\"...")
    startTime = time.time()
    ret = readVocab("vocab.txt")
    if ret < 0:
        print("Failed!! Exiting.")
        print("--"*20)
        exit()
    timeElapsed = time.time() - startTime
    print('Completed in {:.0f}m {:.0f}s'.format((timeElapsed / 60), (timeElapsed % 60)))
    print("--"*20)

    #Read text corpus.
    print("Reading corpus file \"text8\"...")
    startTime = time.time()
    ret = readcorpus("text8")
    if ret < 0:
        print("Failed!! Exiting.")
        print("--"*20)
        exit()
    timeElapsed = time.time() - startTime
    print('Completed in {:.0f}m {:.0f}s'.format((timeElapsed / 60), (timeElapsed % 60)))
    print("--"*20)

    #Generate random matrices as initial weights.
    print("Initializing Neural Net Weights...")
    startTime = time.time()
    w1 = np.random.random((gVocabSize,gHidLyrSize))
    w2 = np.random.random((gHidLyrSize,gVocabSize))
    timeElapsed = time.time() - startTime
    print('Completed in {:.0f}m {:.0f}s'.format((timeElapsed / 60), (timeElapsed % 60)))
    print("--"*20)

    #Initialize unigram table for negative sampling.
    print("Initializing unigram table...")
    startTime = time.time()
    initUnigramTable()
    timeElapsed = time.time() - startTime
    print('Completed in {:.0f}m {:.0f}s'.format((timeElapsed / 60), (timeElapsed % 60)))
    print("--"*20)

    #Get length of training data.
    trainDataLen = len(gTrainData)

    #Run learning epochs.
    print("Running Learning Epochs...")
    print("--"*20)
    for epoch in range(gTotalEpochs):
        print('Epoch {}/{}'.format(epoch, gTotalEpochs - 1))
        startTime = time.time()    #Log start time for epoch.
        #Perform Stochastic Gradient Descent. Iterate over each training example.
        for data in tqdm(range(trainDataLen)):
            #Perform forward propogation.
            targetIndex = gVocabDict[gTrainData[data]]
            contextIndex = gVocabDict[gLabelData[data]]
            hiddenLayer, outputLayer = forwardPropogation(targetIndex, w1, w2)
            #Perform negative sampling to get a random list of words/neg samples.
            nsIndices = negativeSampling()
            #Perform backward propogation on context word and neg samples.
            #This will update the W1 and W2 weight matrices (word vectors).
            w1, w2 = backPropogation(targetIndex, contextIndex, nsIndices, w1, w2, hiddenLayer)
        timeElapsed = time.time() - startTime
        print('Completed in {:.0f}m {:.0f}s'.format(
        (timeElapsed / 60), (timeElapsed % 60)))

    print("All epochs finished.")
    print("--"*20)

    print("Saving word vectors to text file.")
    print("--"*20)

    f = open('vectors.txt', 'w+')   #Open a text file.
    #Sort the vocabulary as per keys, that is, alphabetically.
    sorted_vocab = sorted(gVocabDict.keys())
    list = []
    for word in sorted_vocab:
        line_string = word + " " + ' '.join(map(str, w1[gVocabDict[word]]))
        f.write(line_string + '\n')
    f.close()

    print("Word vector file generated.")
    print("--"*20)
