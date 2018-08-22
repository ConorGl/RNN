# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:55:19 2018

@author: ConorGL

The aim of this project is to look in to Reddit comments and to use them to make our own.
This will form part 1 of a larger project which will be to generate our own Floria Man stories.
"""
import nltk
import csv
import itertools
from datetime import datetime
import numpy as np
from CreateRNN import RNNNumpy
import sys
#nltk.download('punkt')


# Outer SGD Loop
    # - model: The RNN model instance
    # - X_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - nepoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluate_loss_after == 0):
            loss = model.calculate_loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{time}: Loss after num_examples_seen={num_examples_seen} epoch={epoch}: {loss}")
            # Adjust the learning rate if loss increases
            if (len(losses) < 1 and losses[-1][1] < losses[-2][1]):
                learning_rate = learning_rate * 0.5 
                print(f"Setting learning rate to {learning_rate}")
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            # One SGD step
            model.sgd_step(X_train[i], y_train[i], learning_rate)
            num_examples_seen += 1
             
"""
We need to set certain variables here for use in the cleaning of the data set.
The vocabulary_size limit imposed is in order to keep the training time down.
Unknown_token will be used as a subtitue for anything not in these 8000 words.
The sentence_start_token and sentence_end_token will be used for to mark where
the start and end of the sentences are. We will use them later on in such a way
that we will construct our generated sentences with sentence_start_token followed
by a the next most likely more.
"""
vocabulary_size = 8500
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

#Start by reading the data row-by-row, tokenising it sentence by sentence.
#Once we have our massive list of list of sentences we flatten it with
#itertools.chain and add our sentence start/end tokens
print("Reading csv file...")
with open('Tutorial/rnn-tutorial-rnnlm/data/reddit-comments-2015-08.csv', 'r', encoding = 'utf-8') as f:
    reader = csv.reader(f)
    sentences = []
    for row in reader:
        if row:
        # Split full comments into sentences
            sentences.append(nltk.sent_tokenize(row[0].lower()))
sentences = itertools.chain(*sentences)
sentences = ['{} {} {}'.format(sentence_start_token, x, sentence_end_token) for x in sentences]      
print ("Parsed {} sentences.".format(len(sentences)))

# Tokenize the sentences into words. We could use the re package here
# to remove punctionation from the sentences as a possible word save.
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found {} unique words tokens.".format(len(word_freq.items())))
 
# Get the vocabulary_size most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print("Using vocabulary size {}.".format(vocabulary_size))
print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format(vocab[-1][0], vocab[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print("\nExample sentence: '{}'".format(sentences[1000]))
print("\nExample sentence after Pre-processing: '{}'".format(tokenized_sentences[1000]))
 
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

np.random.seed(10)
model = RNNNumpy(vocabulary_size)

"""
We can use the below code in order to a quick 'test' to see what the output of our RNN would be.
It currently only spits out gobbledy-gook....

o, s = model.forward_propagation(X_train[10])
predictions = model.predict(X_train[10])
print(predictions.shape)
print(predictions)

predicted_words = [index_to_word[i] for i in predictions]
predicted_sentence = " ".join(predicted_words)

print("Expected Loss for random predictions: {}".format(np.log(vocabulary_size)))
print("Actual loss:{}".format(model.calculate_loss(X_train[:1000], y_train[:1000])))
"""


"""
This next chunk of code was used to have a quick test that the gradient check f
function will work properly
grad_check_vocab_size = 100
np.random.seed(10)
model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
model.gradient_check([0,1,2,3], [1,2,3,4])
"""

np.random.seed(10)
model = RNNNumpy(vocabulary_size)
%timeit model.numpy_sgd_step(X_train[10], y_train[10], 0.005)

# Train on a small subset of the data to see what happens
model = RNNNumpy(vocabulary_size)
losses = train_with_sgd(model, X_train[:100], y_train[:100], nepoch=10, evaluate_loss_after=1)
