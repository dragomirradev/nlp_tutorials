corpus = ["Apple Orange Orange Apple",\
          "Apple Banana Apple Banana",\
          "Banana Apple Banana Banana Banana Apple",\
          "Banana Orange Banana Banana Orange Banana",\
          "Banana Apple Banana Banana Orange Banana"]
          
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

vectorizer.fit(corpus)

corpus_vec = vectorizer.transform(corpus).toarray()

print(corpus_vec)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()

vectorizer.fit(corpus)

corpus_vec = vectorizer.transform(corpus).toarray()

print(corpus_vec)

import nltk.stem
from nltk.stem import WordNetLemmatizer

class WordNetStemmer(WordNetLemmatizer):
    def stem(self,word,pos=u'n'):
        return self.lemmatize(word,pos)

class Stemmer(object):
    def __init__(self,stemmer_type):
        self.stemmer_type = stemmer_type
        if (self.stemmer_type == 'porter'):
            self.stemmer = nltk.stem.PorterStemmer()
        elif (self.stemmer_type == 'snowball'):
            self.stemmer = nltk.stem.SnowballStemmer('english')
        elif (self.stemmer_type == 'lemmatize'):
            self.stemmer = WordNetStemmer()
        else:
            raise NameError("'"+stemmer_type +"'" + " not supported")

stemmer1 = Stemmer('porter').stemmer
stemmer2 = Stemmer('snowball').stemmer
stemmer3 = Stemmer('lemmatize').stemmer


some_words=['applied', 'cars', 'written', 'done', 'painting']
print("Original:", some_words)

stemmed = [stemmer1.stem(w) for w in some_words]
print("Stemmed with porter:", stemmed)

stemmed = [stemmer2.stem(w) for w in some_words]
print("Stemmed with snowball:",stemmed)

stemmed = [stemmer3.stem(w,'v') for w in some_words]
print("Stemmed with lemmatize:",stemmed)

print('This is a sentence.'.split())

raw = """'When I'M a Duchess,' she said to herself, (not in a very hopeful tone
though), 'I won't have any pepper in my kitchen AT ALL. Soup does very
well without--Maybe it's ! 12$ 82% always pepper $10.2 U.S.A. that makes 
people hot-tempered,'..."""

import re

print(re.split(r' ', raw))

print(re.split(r'\W+', raw))

print(re.findall(r"\b\w\w+\b",raw))

print(re.findall(r"(?:[A-Z]\.)+|\w+(?:[']\w+)*|\$?\d+(?:\.\d+)?%?", raw))














