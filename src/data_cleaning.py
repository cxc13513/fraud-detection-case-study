import pandas as pd
import numpy as np


class DataCleaning(object):

    def __init__(self, df=None):
        self.df = df

    def convert_dates(self):
        date_fields = ['approx_payout_date',
                        'event_created',
                        'event_end',
                        'event_start',
                        'user_created']
        for field in date_fields:
            self.df[field] = pd.to_datetime(self.df[field],unit='s')

    def create_y(self):
        self.df['fraud'] = map(lambda x: 'fraud' in x, self.df.acct_type)


    def save_clean_json(self, path):
        self.df.to_json(path)

if __name__ == "__main__":

    df = pd.read_json("data/raw/data.json")

    dc = DataCleaning(df)

    dc.create_y()

    dc.convert_dates()

    dc.save_clean_json("data/processed/clean_data.json")
