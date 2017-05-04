'''
data_handler class for preparing data for analysis from the Tradeshift Kaggle challenge.

If you want to process the raw data then run this file.  Otherwise, import the file and use the load methods to retrieve the data.

I would recommend installing memory_profiler to keep track of memory usage while fiddling with this code.  In its current state uses < 6GB.  This should work on modern machines, it can also be made leaner by iterating over chunks of the csv or just doing one data set at a time.

Assumes raw data is available in the Data/ folder from the current working directory.

    get_columns(dataframe)   - builds a dictionary of each feature type
    get_hashlist(dataframes) - builds a list of the unique hashes from the full data set
    process_bin(dataframes)  - replace the NO/YES/NaN columns with 0/1/2
    process_hash(dataframes) - replace each hash with its index in the list of unique hashes
    process_data()           - loads the data, runs the above, and records the processed dataframes
    load_columns()           - loads the columns dictionary from file
    load_train()             - loads the train df and its labels from file
    load_cv()                - loads the cv df and its labels from file
    load_test()              - loads the test df from file

Brock Moir
bmoir@ualberta.net
Oct 23, 2014
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import cPickle
import os

class data_handler():

  def __init__(self):
    self.columns = {}   # dictionary of column types
    self.hash_list = [] # list of unique hashes in the data set

    self.datapath = os.getcwd() + '/Data/'
    self.trainpath = self.datapath + 'train.csv'
    self.testpath = self.datapath + 'test.csv'
    self.labelspath = self.datapath + 'trainLabels.csv'
    self.columnspath = self.datapath + 'columns.p'
    self.Xtestpath = self.datapath + 'Xtest.p'
    self.Xtrainpath = self.datapath + 'Xtrain.p' 
    self.Ytrainpath = self.datapath + 'Ytrain.p' 
    self.Xcvpath = self.datapath + 'Xcv.p' 
    self.Ycvpath = self.datapath + 'Ycv.p' 

  @profile
  def get_columns(self, df): # give it a dataframe and it will fill the columns dictionary
    print 'Getting columns'
    self.columns['object'] = df.select_dtypes(['object']).columns.values
    self.columns['float']  = df.select_dtypes(['float64']).columns.values
    self.columns['int']    = df.select_dtypes(['int64']).columns.values[1:]
    self.columns['bin']    = []
    self.columns['hash']   = []
    for column in self.columns['object']:
      if 'YES' in df[column].values:  
        self.columns['bin'].append(column)
      else: self.columns['hash'].append(column) 
    return self.columns

  @profile
  def get_hashlist(self, dfs): # I am pretty sure all of the unique hashes are in the train set
    print 'Getting list of unique hashes'
    for df in dfs:
      self.hash_list = list(set(self.hash_list) | set(pd.unique(df[self.columns['hash']].values.ravel()).tolist()))
    return self.hash_list

  @profile
  def process_int(self, dfs):
    print 'Processing integar columns' 
    for df in dfs:
      for column in self.columns['int']:
        colmin = df[column].min()
        if colmin <= 0: df[column] += colmin + 1
      df.fillna(0)
    return dfs    

  @profile
  def process_bin(self, dfs): # The binary columns are come with either yes, no or missing values
    print 'Processing binary columns'
    mapdict = {np.nan:2, 'YES':1, 'NO':0}
    for df in dfs:
      for column in self.columns['bin']:
        df[column] = df[column].map(mapdict)
    return dfs

  @profile
  def process_hash(self, dfs): # There are over one million unique hashes
    print 'Processing hash columns'
    for df in dfs:
      mapdict = dict(zip(self.hash_list, range(len(self.hash_list))))
      print len(self.hash_list), 'Unique hashes'   
      for column in self.columns['hash']:
        df[column] = df[column].map(mapdict)
    return dfs

  @profile
  def trainsplit(self, train, labels):
    print 'Splitting the training set'
    xcolumns = train.columns
    ycolumns = labels.columns
    train, cv, labels, cvlabels = train_test_split(train, labels, random_state=42)
    return pd.DataFrame(train, columns=xcolumns), pd.DataFrame(labels, columns=ycolumns), pd.DataFrame(cv, columns=xcolumns), pd.DataFrame(cvlabels, columns=ycolumns) 

  @profile
  def process_data(self): # Processes the data, and records it for easy loading later
    traindf, testdf = pd.read_csv(self.trainpath), pd.read_csv(self.testpath)    
    cPickle.dump(self.get_columns(testdf), open(self.columnspath, 'w'))
    self.get_hashlist([traindf,testdf])
    traindf, testdf = self.process_hash(self.process_bin(self.process_int([traindf,testdf])))
    testdf.to_pickle(self.Xtestpath)
    traindf, labelsdf, cvdf, cvlabelsdf = self.trainsplit(traindf, pd.read_csv(self.labelspath))
    traindf.to_pickle(self.Xtrainpath)
    labelsdf.to_pickle(self.Ytrainpath) 
    cvdf.to_pickle(self.Xcvpath)
    cvlabelsdf.to_pickle(self.Ycvpath) 

  def load_columns(self):
    return cPickle.load(open(self.columnspath, 'r'))

  def load_test(self):
    return pd.read_pickle(self.Xtestpath)

  def load_train(self):
    return pd.read_pickle(self.Xtrainpath), pd.read_pickle(self.Ytrainpath)

  def load_cv(self):
    return pd.read_pickle(self.Xcvpath), pd.read_pickle(self.Ycvpath)

if __name__=='__main__':
  processor = data_handler()
  processor.process_data()

