import pandas as pd
from data_clean import DataCleaning
from features import run_all
import numpy as np
# from sklearn.model_selection import train_test_split

class PrepareTrain(object):

    '''
    Creates a train class to prepare the train data
    '''
    def __init__(self,df,logistic=False, undersample=False):
        self.df = df
        self.logistic = logistic
        self.undersample_condition = undersample

    def undersample(self):
        df_non_fraud = self.df[self.df.fraud == False]
        import pdb; pdb.set_trace()
        undersampled_indices = np.random.choice(np.array(df_non_fraud.index),int(0.5*len(df_non_fraud)))
        self.df = self.df.drop(undersampled_indices,axis=0)

    def clean_train(self):
        self.df = DataCleaning(self.df).convert_dates()

    def featurize(self):
        X = run_all(self.df,logistic=self.logistic)
        return X

    def create_y(self):
        self.df['fraud'] = map(lambda x: 'fraud' in x, self.df.acct_type)

    def prepare_data(self):
        '''
        Run general cleaning steps on DataFrame, return X and y
        '''
        # self.clean_train()
        self.create_y()
        if self.undersample_condition:
            self.undersample()
        y = self.df.pop('fraud')
        X = self.featurize()
        return X,y
