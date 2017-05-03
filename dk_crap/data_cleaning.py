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

    def make_pop_email(self):
        self.df['pop_email'] = map(lambda x: 1 if x in ['hotmail.com', 'gmail.com','yahoo.com', 'aol.com'] else 0, self.df.email_domain)

    def save_clean_json(self, path):
        self.df.to_json(path)

    def make_avg_ticket_price(self):
        self.df['avg_ticket_price'] = [np.mean([y['cost'] for y in x]) for x in self.df.ticket_types]



if __name__ == "__main__":

    # read raw json data
    df = pd.read_json("data/raw/data.json")

    # init DataCleaning class
    dc = DataCleaning(df)

    # clanining steps
    dc.convert_dates()

    # make features
    dc.make_pop_email()

    # create y labels and save the clean file
    dc.create_y()
    dc.save_clean_json("data/processed/clean_data.json")
