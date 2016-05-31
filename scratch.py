import numpy as np
import pandas as pd
import re 

from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from utils import Timer

from nltk.book import *

import nltk.corpus

from nltk.corpus import wordnet as wn

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None

def stem(x,pos):
    return WordNetLemmatizer().lemmatize(x,pos)

text = word_tokenize("and or we love computer practise is laplace")
nltk.pos_tag(text)


def content_fraction(text):
     stopwords = nltk.corpus.stopwords.words('english')
     content = [w for w in text if w.lower() not in stopwords]
     return len(content) / len(text)

content_fraction(nltk.corpus.reuters.words())


names = nltk.corpus.names
names.fileids()

male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]


wn.synsets('motorcar')

wn.synset('car.n.01').lemma_names()

wn.synset('car.n.01').definition()

wn.synset('car.n.01').examples()

stopwords.words('english')



token = word_tokenize(raw)
text = nltk.Text(token)

fdist = FreqDist(text)

fdist.most_common(50)

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's ! 12$ 82% always pepper $10.2 U.S.A. that makes people hot-tempered,'..."""

print(re.split(r' ', raw))

re.split(r'[ \t\n]+', raw)

re.split(r'\W+', raw)

re.findall(r'\w+|\S\w*', raw)

re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\S\w*", raw)

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer("[\w']+")
tokenizer.tokenize(raw)

pattern=r"(?:[A-Z]\.)+|\w+(?:[']\w+)*|\$?\d+(?:\.\d+)?%?"
tokenizer = RegexpTokenizer(pattern)
tokenizer.tokenize(raw)

pattern = r"(?u)\b\w\w+\b"
tokenizer = RegexpTokenizer(pattern)

tokenizer.tokenize(raw)

import timeit
%timeit tokens1 = tokenizer.tokenize(raw.lower())

pattern = re.compile(pattern, flags = re.UNICODE | re.LOCALE)

%timeit tokens2 = [x for x in pattern.findall(raw.lower())]



tokenizer = RegexpTokenizer(pattern)
%timeit tokenizer.tokenize(raw)



from sklearn.feature_extraction.text import CountVectorizer

corpus = [ \
    'This apple is banana the  that first document.', \
    'This mango is the  that second second document.', \
    'And this and banana mango that the third one.', \
    'Is orange this that the first document?', \
]
 
vectorizer = CountVectorizer(min_df=1)

X = vectorizer.fit_transform(corpus)

vectorizer.get_feature_names()

X.toarray()   

vectorizer.transform(['Something completely new this.']).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(min_df=1)

vectorizer.fit(corpus)

# http://radimrehurek.com/gensim/tut1.html
documents = ["Human machine interface for lab abc computer applications",\
             "A survey of user opinion of computer system response time",\
             "The EPS user interface management system",\
             "System and human system engineering testing of EPS",\
             "Relation of user perceived response time to error measurement",\
             "The generation of random binary unordered trees",\
             "The intersection graph of paths in trees",\
             "Graph minors IV Widths of trees and well quasi ordering",\
             "Graph minors A survey"]
             
from sklearn.datasets import fetch_20newsgroups
newsgroups_train = fetch_20newsgroups(subset='train')

print(newsgroups_train.target_names)

filenames = newsgroups_train.filenames
target = newsgroups_train.target
data = newsgroups_train.data

import requests
url = "http://mattmahoney.net/dc/text8.zip"
r = requests.get(url)
print(len(r.content))


print(len(data))
print(data[10])

print(set(target))

file='/Users/arman/word2vec-mac/text8'

with open(file,'r') as f:
    text8 = f.read()
    
print(text8[0:1000])

from gensim.models import Word2Vec

with Timer("Training gensim model"):
    model = Word2Vec(text8,size=300, window=5, workers=4)

#[Training gensim model] Elapsed: 0hour:19min:59sec


file='/Users/arman/word2vec-mac/trained_model.bin'

#model.save(file)

file2='/Users/arman/word2vec-mac/vectors.bin'

import numpy as np
from pprint import pprint


model = Word2Vec.load_word2vec_format(file2, binary=True)

pprint(model.most_similar(positive=['woman', 'king'], negative=['man']))

model.most_similar(positive=['italy', 'paris'], negative=['rome'])

model.most_similar(positive=['grandfather','mother'],negative=['father'])

model.most_similar(positive=['night', 'sun'], negative=['day'])

model.most_similar(positive=['air', 'car'], negative=['street'])

model.most_similar(positive=['small','cold'],negative=['large'])

model.most_similar(positive=['art','experiment'],negative=['science'])

model.most_similar(positive=['men','car'],negative=['man'])

model.most_similar(positive=['great','mathematics'],negative=['good'])

model.most_similar(positive=['blue','friends'],negative=['colors'])

####

model.doesnt_match(['breakfast', 'cereal', 'dinner', 'lunch'])

model.doesnt_match('car truck house bicycle'.split())

model.doesnt_match('soda juice coke beer hamburger'.split())


newdoc = [ \
  'apple this', \
  'mango', \
  'banana', \
  'this'
]

target = np.array(vectorizer.transform(newdoc).toarray())

org = np.array(vectorizer.transform(corpus).toarray())


np.dot(org,target.T)

from sklearn import preprocessing

lb = preprocessing.LabelBinarizer()
a = lb.fit_transform([0,1,2,1,2,2,3,0,1,0,0])
a[:,:-1]
