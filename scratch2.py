
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



















