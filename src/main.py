import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_clean import DataCleaning
import features as features
reload (features)

def clean_all(data_clean_object,train = False,logistic=False):
    if train:
        X, y = data_clean_object.prepare_data(train=train)
    else:
        X = data_clean_object.prepare_data(train=train)
    column_list = ['body_length','currency',
                    'description','email_domain','event_created',
                    'event_end','event_published', 'user_created',
                    'name','num_order','num_payouts','has_logo',
                    'org_desc']
    X = features.run_all(X,column_list)
    if train:
        return X,y
    return X

if __name__ == "__main__":

    df = pd.read_json("data/raw/data.json")
    X_train, X_test = train_test_split(df,test_size = 0.2, random_state = 1)
    dc = DataCleaning(X_train)
    # X, y = dc.prepare_data()
    # X = features.minimal_df(X,['country','description'])
    X,y = clean_all(dc,train=True)

    # dc.create_y()
    #
    # dc.convert_dates()

    # dc.save_clean_json("data/processed/clean_data.json")
