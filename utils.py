# -*- coding: utf-8 -*-
"""
Implements some helper functions / classes 

__file__

    utils.py
    
__description__

    - tic tac class for timing
    - overwrites print function with default flushing set to True
    - dot progress functions
    
__author__

    Arman Akbarian
    
"""
import sys
import builtins
import time
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict

#### implementation of tic tac for timing ####

class Timer(object):
    """
    Implements popular tic tac functions similar to Matlab
    
    Example:
    --------
    
    1- Using tic and tac methods:
    
       >>> from utils import Timer
       >>> timer = Timer()
       >>> timer.tic()
       >>> do_some_stuff()
       >>> timer.tac()
       Elapsed: 0hour:14min:12sec
       
    2- Using ``with`` block:
    
       >>> from utils import Timer
       >>> with Timer('Some progress block name'):
       ...      do_some_stuff()
       [Some progress block name] Elapsed: 0hour:0min:10sec
    """
    def __init__(self, name=None,output='default'):
        self.name = name
        self.output = output

    def __enter__(self):
        self.tic()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[{}] '.format(self.name),end='')
        self.tac()
            
    def tic(self):
        self._start_time = time.time()
        
    def tac(self):
        t_sec = round(time.time() - self._start_time)
        if self.output == 'sec':
            print('Elapsed: {}sec'.format(t_sec))
            return
        (t_min, t_sec) = divmod(t_sec,60)
        if self.output == 'min':
            print('Elapsed: {}min:{:0>2}sec'.format(t_min,t_sec))
            return  
        (t_hour,t_min) = divmod(t_min,60) 
        print('Elapsed: {}hour:{:0>2}min:{:0>2}sec'.format(t_hour,t_min,t_sec))
        
class KNeighboorsClassifier(object):
    """
    Implements a NN using KDTree, since scikit learn's NN classifier
    is way too slow! and also only returns 1 category!
    
    Example:
    --------
    
    1- Building from a given train set:
       >>> from utils import KNeighboorsClassifier
       >>> X = some_function_that_gets_data_matrix(dfTrain)
       >>> y = some_function_that_gets_target_vector(dfTrain)
       >>> clf = KNeighboorsClassifier(n_neighboors=10,n_classes=2)
       >>> clf.fit(X,y)
       >>> X_test = SFTG_test_data_matrix(dfTest)
       >>> result = clf.predict(X_test)
       
    2- Using a pre-existig kdtree:
       >>> clf = KNeighboorsClassifier(n_neighboors=10,n_classes=2,tree=myKDTree,y=target)
    
    Parameters:
    -----------
    
     n_neighboors: default number of neighboors to use in .fit method
     
     n_classes: default number of classes that .fit returns 
                n_classes = 1, will return the standard kNN highest mode 
                class id
    
     tree: a KDTree to use (won't need fitting anymore, if tree is given)
     
     y: (type: np.array) the vector of target classes
                
     chunk_size: number of data points to use in each query and finding of
                 the kNN. If set to < 1 will use chunk_size as the ratio
                 for splitting the entire X vector.
                 *Note*: increasing this number will increase the memory usage                 
     
     verbose: if True, will return a progress bar of process
     
    """
    def __init__(self,n_neighboors=100,n_classes =3,tree=None,y=None):
        self.n_neighboors = n_neighboors
        self.tree = tree
        self.n_classes = n_classes
        self.y = np.array(y)
        
    def fit(self,X,y):
        if self.tree is None:
            self.tree = KDTree(X)
        if self.y is None:
            self.y = np.array(y)
    
    def predict(self,X,n_neighboors='default',n_classes='default',\
                chunk_size=10000,verbose=True):
        
        
        if (self.tree is None):
            print("Must call fit first!")
            return None
            
        if (self.y is None):
            print("Target value is not given, call fit first!")
            return None
            
        if n_neighboors == 'default':
            n_neighboors = self.n_neighboors
        
        if n_classes == 'default':
            n_classes = self.n_classes
        
        i_ = 0
        X_size_ = len(X)
        top_choices = list()
        ############################################################################
        ## can use chuck_size as:
        ##   1) integer -> 10000 (will query and process 10000 points at each batch)
        ##   2) ratio -> 0.1 (will query and process 10% of data at a each batch
        ## the choice is indicated with if chunc_size < 1 or not
        ############################################################################
        if (chunk_size < 1):
            chunk_size = round(chunk_size*X_size_)
        
        if (verbose):
                sys.stdout.write('\r')
                sys.stdout.write("[%-50s] %d%% complete" % ('='*(0), 0))  
                sys.stdout.flush()
        while (i_ < X_size_):
            
            start_ = i_
            end_ = i_ + chunk_size
            if (end_ > X_size_):
                end_ = X_size_
            
            indx_ = self.tree.query(X[start_:end_], k=n_neighboors,\
                               return_distance=False)
        
          
            for item_ in indx_:
                ys_ = self.y[item_]
                uniques_ , counts_ = np.unique(ys_, return_index=False,\
                                               return_counts=True)  
                sorted_args_ = np.argsort(-counts_)[:n_classes]
                occur_count_top_n_class =  np.concatenate( (uniques_[sorted_args_],\
                                                           counts_[sorted_args_]))                              
                top_choices.append(list(occur_count_top_n_class))
                
            i_ += chunk_size   
            
            if (verbose):
                sys.stdout.write('\r')
                perc = round(end_/X_size_*100)
                sys.stdout.write("[%-50s] %d%% complete" % ('='*(perc//2), perc))
                sys.stdout.flush()
                
        if (verbose):    
            sys.stdout.write('\n')
            
        return top_choices
        
        
        
        
        
        
        
#############################
###  ROUTINES FOR TESTING ###
#############################
        
        
###### uses panda's series, for testing, insanely slow! ########        
def create_top_list_batch(batch_start,batch_stop,pid,indx,top_l):
    top_choices = list()
    for i in range(batch_start,batch_stop):
        xx = (pid[indx[i]]).value_counts()
        aa = np.concatenate( (np.array(xx.index[:top_l]), np.array(xx[:top_l]) ) )
        top_choices.append(list(aa))
    return top_choices  

#### non_batch version of above, slow as hell! #######
def create_top_list(pid,indx,top_l):
    top_choices = list()
    for i in range(len(indx)):
        xx = (pid[indx[i]]).value_counts()
        aa = np.concatenate( (np.array(xx.index[:top_l]), np.array(xx[:top_l]) ) )
        top_choices.append(list(aa))
    return top_choices

#### implementation using int dictionary, fast, doesn't return counts #####
def find_modes(indx,pid,top_l):
    modes = list()
    for i in range(len(indx)):
        d = defaultdict(int)
        for j in indx[i]:
            d[pid[j]] += 1
            #d[pid[j]] = d.get(pid[j],0) + 1
        modes.append(sorted(d,key=d.get,reverse=True)[:top_l])   
    return modes

#### this is the fastest! returns counts #####
def find_modes_np(indx,pid,top_l):
    modes = list()
    for item in indx:
        xx = pid[item]
        item_unique , item_count = np.unique(xx, return_index=False,return_counts=True)   
        sorted_args = np.argsort(-item_count)[:top_l]
        aa_ = np.concatenate( (item_unique[sorted_args],item_count[sorted_args]))
        modes.append(list(aa_))
    return modes
        
#### for progress display ####
  
def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    builtins.print(*objects, sep=sep, end=end, file=file, flush=flush)

def prog():
    print(".",end='')

def longprog():
    print("....",end='')

        
if __name__ == '__main__':
    
    print("=== Using timer with `.tic()` and `.tac()` ===")
    timer = Timer()
    timer.tic()
    time.sleep(7)
    timer.tac()
    
    print("=== Using Timer via `with` block ===")
    with Timer('my code chunk name'):
        time.sleep(9)
    
        