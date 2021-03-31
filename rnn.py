import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
from pathlib import Path 
import time
from tqdm import tqdm
from data_loader import fetch_data
from gensim.models import Word2Vec
import csv

unk = '<UNK>'

def read_data(file):
    with open(file) as f:
        lines = csv.reader(f)
        next(lines)
        stories = []
        labels = []
        for line in lines:
            paragraph = []
            paragraph.append(line[1])
            paragraph.append(line[2])
            paragraph.append(line[3])
            paragraph.append(line[4])
            paragraph.append(line[5])
            paragraph.append(line[6])
            stories.append(paragraph)
            labels.append(line[7])
    return stories, labels

class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()

        self.h = h
        # self.hidden = torch.zeros(1,1,self.h)
        # self.Wxh = nn.Linear(input_dim, h)
        # self.Whh = nn.Linear(h, h)
        self.Why = nn.Linear(h, 2)
        self.activation = nn.ReLU()  # The rectified linear unit; one valid choice of activation function
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer)
        # Fill in relevant parameters
        # Ensure parameters are initialized to small values, see PyTorch documentation for guidance
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs, hidden):
        # begin code

        # hidden = self.Whh(hidden) + self.Wxh(inputs)
        # output = self.Why(hidden)

        _, hidden = self.rnn(inputs, hidden)
        output = self.Why(hidden)
        output = output.sum(dim=0)
        predicted_vector = self.softmax(output)
        return predicted_vector

#Create unigram and bigram counts for the training data
def create_bigram(stories):
    data = []
    for paragraph in stories:
        for i in range(4):
            for word in paragraph[i].split():
                data.append(word)
                
    listOfBigrams = []
    bigramCounts = {}
    unigramCounts = {}

    for i in range(len(data)):
        if i < len(data) - 1:

            if (data[i], data[i+1]) in bigramCounts:
                bigramCounts[(data[i], data[i + 1])] += 1
                listOfBigrams.append((data[i], data[i + 1]))
            else:
                bigramCounts[(data[i], data[i + 1])] = 0
                listOfBigrams.append((unk, unk))

        if data[i] in unigramCounts:
            unigramCounts[data[i]] += 1
        else:
            unigramCounts[data[i]] = 0
    
    #handling unknown words by replacing the first occurence of each unigram/bigram with <UNK>
    unigramSize = len(unigramCounts)
    unigramCounts[unk] = unigramSize
    bigramSize = len(bigramCounts)
    bigramCounts[(unk, unk)] = bigramSize
    
    return listOfBigrams, unigramCounts, bigramCounts

#Calculate the probability of N-gram model without smoothing
def calcProb(listOfBigrams, unigramCounts, bigramCounts):
    listOfUnigramProb={}
    for unigram in unigramCounts:
        listOfUnigramProb[unigram] = (unigramCounts[unigram]) / (len(unigramCounts))
        
    listOfBigramProb = {}
    for bigram in listOfBigrams:
        word1 = bigram[0]
		
        listOfBigramProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))

    return listOfUnigramProb, listOfBigramProb

#Add-k smoothing
def additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, k):
	listOfProb = {}

	for bigram in listOfBigrams:
		word1 = bigram[0]
		listOfProb[bigram] = (bigramCounts[bigram] + k)/(unigramCounts[word1] + len(unigramCounts) * k)

	return listOfProb

#Perplexity function
def perplexity(listOfProb, ending):
    data = ending.split()
    power = 0
    
    for i in range(len(data) - 1):
        try:
            prob = listOfProb[(data[i], data[i + 1])]
        except KeyError:
            prob = listOfProb[(unk, unk)]
        power -= math.log(prob)
        
    power = power / len(data)

    return float(math.exp(power))

#Vectorize the training data
def convert_to_vector_representation(stories, labels, list_of_prob):
    sentences = []
    for paragraph in stories:
        for sentence in paragraph:
            sentences.append(sentence.split())
    model = Word2Vec(sentences, min_count=1)
    
    vectorized_data = []
    for paragraph, label in zip(stories, labels):
        label = int(label)
        correct = []
        wrong = []
        for i in range(4):
            for word in paragraph[i].split():
                correct.append(model.wv[word])
                wrong.append(model.wv[word])
        for word in paragraph[3 + label].split():
            correct.append(model.wv[word])
        for word in paragraph[6 - label].split():
            wrong.append(model.wv[word])
        perp_of_correct = perplexity(list_of_prob, paragraph[3 + label])
        perp_of_wrong = perplexity(list_of_prob, paragraph[6 - label])
        perp_vector_correct = np.zeros(100)
        perp_vector_wrong = np.zeros(100)
        perp_vector_correct[0] = perp_of_correct
        perp_vector_wrong[0] = perp_of_wrong
        length_vector_correct = np.zeros(100)
        length_vector_wrong = np.zeros(100)
        length_vector_correct[0] = len(paragraph[3 + label])
        length_vector_wrong[0] = len(paragraph[6 - label])
        correct.append(perp_vector_correct)
        correct.append(length_vector_correct)
        wrong.append(perp_vector_wrong)
        wrong.append(length_vector_wrong)
        vectorized_data.append((correct, 1))
        vectorized_data.append((wrong, 0))
    return vectorized_data
       

'''
# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(stories):
    vocab = set()
    for paragraph in stories:
        for i in range(4):
            for word in paragraph[i].split():
                vocab.add(word)
    return vocab 

# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
	vocab_list = sorted(vocab)
	vocab_list.append(unk)
	word2index = {}
	index2word = {}
	for index, word in enumerate(vocab_list):
		word2index[word] = index 
		index2word[index] = word 
	vocab.add(unk)
	return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(stories, labels, word2index, listOfProb):
    vectorized_data = []
    for paragraph, y in zip(stories, labels):
        vector = torch.zeros(len(word2index) + 2) 
        for i in range(4):
            for word in paragraph[i].split():
                index = word2index.get(word, word2index[unk])
                vector[index] += 1
        for i in range(4, 6):
            perp = perplexity(listOfProb, paragraph[i])
            vector[i - 6] = perp
        vectorized_data.append((vector, y))
    return vectorized_data
'''
    

def main(hidden_dim, number_of_epochs): # Add relevant parameters
    '''
	print("Fetching data")
	train_data, valid_data = fetch_data() # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
	vocab = make_vocab(train_data)
	vocab, word2index, index2word = make_indices(vocab)
	print("Fetched and indexed data")
	train_data = convert_to_vector_representation(train_data, word2index)
	valid_data = convert_to_vector_representation(valid_data, word2index)
	print("Vectorized data")
    '''
    train_stories, train_labels = read_data("train.csv")
    listOfBigrams, unigramCounts, bigramCounts = create_bigram(train_stories)
    listOfProb = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, 1)
    #perp = perplexity(listOfProb, "went next door to return it")
    #print(type(perp))
    train_data = convert_to_vector_representation(train_stories, train_labels, listOfProb)
	# Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
	# Further, think about where the vectors will come from. There are 3 reasonable choices:
	# 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
	# 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
	# 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further. 
	# Option 3 will be the most time consuming, so we do not recommend starting with this
   
    model = RNN(100, hidden_dim) # Fill in parameters
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) 
    print("Training for {} epochs".format(number_of_epochs))
    for epoch in range(number_of_epochs): # How will you decide to stop training and why
		# You will need further code to operationalize training, ffnn.py may be helpful
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16 
        N = len(train_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                hidden = torch.zeros(model.numOfLayer, 1, hidden_dim)
                predicted_vector = model(torch.tensor(input_vector, dtype=torch.float32).view(len(input_vector), 1, -1), hidden)
                #print(predicted_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                #print("predicted_label:", predicted_label)
                #print("gold_label:", gold_label)
                #return
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                #print(example_loss)
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
            #return
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        valid_stories, valid_labels = read_data("dev.csv")
        listOfBigrams, unigramCounts, bigramCounts = create_bigram(valid_stories)
        listOfProb = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, 1)
        valid_data = convert_to_vector_representation(valid_stories, valid_labels, listOfProb)
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        random.shuffle(valid_data) # Good practice to shuffle order of validation data
        minibatch_size = 16 
        N = len(valid_data) 
        for minibatch_index in tqdm(range(N // minibatch_size)):
            #optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                hidden = torch.zeros(model.numOfLayer, 1, hidden_dim)
                predicted_vector = model(torch.tensor(input_vector, dtype=torch.float32).view(len(input_vector), 1, -1), hidden)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            #loss = loss / minibatch_size
            #loss.backward()
            #optimizer.step()
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        # You may find it beneficial to keep track of training accuracy or training loss; 

		# Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance

		# You will need to validate your model. All results for Part 3 should be reported on the validation set. 
		# Consider ffnn.py; making changes to validation if you find them necessary

