import numpy as np
import pandas as pd
import re 

from nltk import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

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

re.split(r' ', raw)

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

import timeit
%timeit tokens1 = tokenizer.tokenize(raw.lower())

pattern = re.compile(pattern, flags = re.UNICODE | re.LOCALE)

%timeit tokens2 = [x for x in pattern.findall(raw.lower())]



tokenizer = RegexpTokenizer(pattern)
tokenizer.tokenize(raw)

for i in xrange(0,10):
    print(i)



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
