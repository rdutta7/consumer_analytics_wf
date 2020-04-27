import numpy as np 
import pandas as pd
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/consumeranalytics/complaints-wf-updated.csv')
data

# Taking a subset of data
verbatim_product = data[['consumer_complaint_narrative','product']]
verbatim_product.head(5)
filtered_verbatim = verbatim_product.dropna()
filtered_verbatim.head(2)

len(filtered_verbatim.consumer_complaint_narrative)
filtered_verbatim['product'].value_counts()
filtered_verbatim['product'].value_counts().plot(kind='bar')

# selected a single complaint to start doing some NLP on
complaint = filtered_verbatim.iloc[1]['consumer_complaint_narrative']
pd.options.display.max_colwidth = 1000
print(complaint)

import spacy #for our NLP processing
import nltk #to use the stopwords library
import string # for a list of all punctuation
from nltk.corpus import stopwords # for a list of stopwords

nlp = spacy.load('en_core_web_sm')
text = nlp(complaint)
text
tokens = [tok for tok in text]
tokens.head(5)
tokens = [tok.lemma_ for tok in text]
tokens
tokens = [tok.lemma_.lower().strip() for tok in text]
tokens
tokens = [tok.lemma_.lower().strip() for tok in text if tok.lemma_ != '-PRON-']
tokens

stop_words = stopwords.words('english')
punctuations = string.punctuation
stop_words

tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
tokens

def cleanup_text(complaint):
    doc = nlp(complaint, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
    return tokens

limit = 100
doc_sample = filtered_verbatim.consumer_complaint_narrative
print('tokenized and lemmatized document: ')

for idx, complaint in enumerate(doc_sample):
    print(cleanup_text(complaint))
    if idx == limit:
        break

doc_sample = doc_sample[0:10000]
processed_docs = doc_sample.map(cleanup_text)

# Bag of Words
import gensim
dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# example of a bag-of-words format
bow_doc_4310 = bow_corpus[4310]

for i in range(len(bow_doc_4310)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_4310[i][0], 
                                                     dictionary[bow_doc_4310[i][0]], 
                                                     bow_doc_4310[i][1]))

# LDA
# Latent Dirichlet allocation (LDA), is an unsupervised algorithm: only the words in the documents are modeled.
# The goal is to infer topics that maximize the likelihood (or the posterior probability) of the collection.
# The LDA algorithm has a number of parameters than can be used to calibrate the output:
# num_topics: In this example we have prescribed a number 10, in a previous run without a prescribed number, the LDA produced 99 clusters which is not very informative for our usecase
# id2word: The previously defined dictionary mapping from word IDs to Words
# Workers: for parralelisation
# chunksize: number of documents to use in each training chunk
# passes: no. passes through the corpus during training
# alpha: Can be set to an 1D array of length equal to the number of expected topics that expresses our a-priori belief for the each topics’ probability.
# decay: A number between (0.5, 1] to weight what percentage of the previous lambda value is forgotten when each new document is examined.
# iterations: Maximum number of iterations through the corpus when inferring the topic distribution of a corpus.
# gamma_threshold: Minimum change in the value of the gamma parameters to continue iterating.
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=7, id2word=dictionary, passes=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# use pyLDA vis to inspect outputs in a more interactive way
import pyLDAvis
import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(lda_model, bow_corpus, dictionary)
pyLDAvis.display(vis_data)
