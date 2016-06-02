
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

















