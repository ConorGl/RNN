import nltk
import csv
import itertools

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"
 # Read the data and append SENTENCE_START and SENTENCE_END tokens

with open('data/reddit-comments-2015-08.csv',encoding = 'utf-8') as f:
    reader = csv.reader(f)
    sentences = []
    for row in reader:
        if row:
        # Split full comments into sentences
            sentences.append(nltk.sent_tokenize(row[0].lower()))
        # Append SENTENCE_START and SENTENCE_END
    sentences = ['{} {} {}'.format(sentence_start_token, x, sentence_end_token) for x in sentences]      
print ("Parsed {} sentences.".format(len(sentences)))
     
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print ("Found {} unique words tokens.".format(len(word_freq.items())))
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size-1)
index_to_word = [x[0] for x in vocab]
index_to_word.append(unknown_token)
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
 
print("Using vocabulary size {}.".format(vocabulary_size))
print("The least frequent word in our vocabulary is '{}' and appeared {} times.".format((vocab[-1][0], vocab[-1][1])))
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
 
# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])
He