import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataCleaning(object):

    def __init__(self, df=None):
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

    def create_y(self):
        self.df['fraud'] = map(lambda x: 'fraud' in x, self.df.acct_type)

    def undersample(self):
        df_non_fraud = self.df[self.df.fraud == False]
        undersampled_indices = np.random.choice(np.array(df_non_fraud.index),int(0.5*len(df_non_fraud)))
        self.df = self.df.drop(undersampled_indices)

    def prepare_data(self,train=False):
        '''
        Run general cleaning steps on DataFrame, return X and y
        '''
        self.create_y()
        self.convert_dates()
        self.undersample()
        if train:
            y = self.df.pop('fraud')
            X = self.df
            return X,y
        return self.df
    #
    # def save_clean_json(self, path):
    #     self.df.to_json(path)



# if __name__ == "__main__":
#
#     df = pd.read_json("data/raw/data.json")
#
#     dc = DataCleaning(df)
#
#     dc.create_y()
#
#     dc.convert_dates()
#
#     dc.email_domains_to_ints()
#
#     dc.save_clean_json("data/processed/clean_data.json")
# =======
#             self.df.to_json(path)
# >>>>>>> Stashed changes
