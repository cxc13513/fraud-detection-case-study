import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCleaning(object):

    def __init__(self, df):
        self.df = df

    def convert_dates(self):
        '''
        Convert dates to Datetime format
        '''
        date_fields = ['approx_payout_date',
                        'event_created',
                        'event_end',
                        'event_start',
                        'user_created']
        for field in date_fields:
            self.df[field] = pd.to_datetime(self.df[field],unit='s')
        return self.df

    def impute_vals(self,col_name,val):
        '''
        Impute defined value val into col_name. Operates over one column at a time
        To run outside of class
        '''
        self.df[col_name] = self.df[col_name].fillna(val)
        return self.df
