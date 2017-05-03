import cPickle as pickle
import pandas as pd
import numpy as np

from db_connect import DBConnector
from data_clean import DataCleaning
import features as features
from model_pipeline import ModelPipeline


def clean_all(data_clean_object,train = False,logistic=False):
    if train:

        X, y = data_clean_object.prepare_data(train=train)
    else:
        X = data_clean_object.prepare_data(train=train)
    column_list = ['body_length','currency',
                    'email_domain','name','num_order','num_payouts','has_logo',
                    'event_created','user_created','country','user_age']
    # column_list = ['body_length','currency',
    #                 'description','email_domain','event_created',
    #                 'event_end','event_published', 'user_created',
    #                 'name','num_order','num_payouts','has_logo',
    #                 'org_desc']
    X = features.run_all(X,column_list)
    if train:
        return X,y
    return X

if __name__ == "__main__":

    # df = pd.read_json("data/raw/example.json")

    df = pd.read_json("data/raw/data.json")
    X = df.sample(n=1)

    dc = DataCleaning(X)
    X,y = clean_all(dc,train=True)

    with open("../data/model/model.pkl") as f:
        model = pickle.load(f)

    y = model.predict(X)

    db = DBConnector()

    db.save_to_db(X,y)
