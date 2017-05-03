from pymongo import MongoClient
import pandas as pd
import json

class DBConnector(object):

    def __init__(self):
        self.c = MongoClient()
        self.db = self.c['fraud_db']
        self.tb = self.db['predictions']

    def save_to_db(self,df,y):
        df['fraud'] = y
        self.tb.insert(df.to_dict('records'))

    def read_frm_db(self):
        return pd.DataFrame(list(self.tb.find()))

if __name__ == "__main__":

    # df = pd.read_json("data/raw/data.json")

    db = DBConnector()

    # db.save_to_db(df,True)
