
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from os.path import join
from bs4 import BeautifulSoup


def review_to_wordlist(review, remove_stopwords=False, split=False):
    """
    Simple text cleaning function, 
    uses BeautifulSoup to extract text content from html
    removes all non-alphabet
    converts to lower case
    can remove stopwords
    can perform simple tokenization using split by whitespace
    """
        
    review_text = BeautifulSoup(review, 'lxml').get_text()
    
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    
    if split:      
        return(words)
    else:
        return(' '.join(words))

### change this line ###
root_dir = '/Users/arman/kaggledata/popcorn'

dfTrain = pd.read_csv(join(root_dir,'labeledTrainData.tsv'),header=0,\
                    delimiter="\t",quoting=3)

dfTest = pd.read_csv(join(root_dir,'testData.tsv'), header=0,\
                   delimiter="\t", quoting=3 )
                   
target = dfTrain['sentiment']


dfTrain['review'] =  dfTrain['review'].map(review_to_wordlist)
dfTest['review'] =  dfTest['review'].map(review_to_wordlist)

train_len = len(dfTrain)


corpus = list(dfTrain['review']) + list(dfTest['review'])

              
tfv = TfidfVectorizer(min_df=3,  max_features=None, ngram_range=(1, 2),\
                      use_idf=True,smooth_idf=True,sublinear_tf=True,\
                      stop_words = 'english')

    
tfv.fit(corpus)

X_all = tfv.transform(corpus)

print(X_all.shape)

train = X_all[:train_len]
test = X_all[train_len:]

#### Parameter tuning using cross-validation #####

Cs = [1,3,10,30,100,300]
for c in Cs:
    clf = LogisticRegression(penalty='l2', dual=True, tol=0.0001,\
                         C=c, fit_intercept=True, intercept_scaling=1.0,\
                         class_weight=None, random_state=None)
                         
    print("c: ",c,"score: ", np.mean(cross_val_score(clf, train, target,\
                            cv=5, scoring='roc_auc')))

#### Best model #####

clf = LogisticRegression(penalty='l2', dual=True, tol=0.0001,\
                         C=30, fit_intercept=True, intercept_scaling=1.0,\
                         class_weight=None, random_state=None)

clf.fit(train,target)

preds = clf.predict_proba(test)[:,1]
dfOut = pd.DataFrame( data={"id":dfTest["id"], "sentiment":preds} )
dfOut.to_csv(join(root_dir,'submission.csv'), index=False, quoting=3)



