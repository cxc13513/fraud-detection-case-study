import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from data_clean import DataCleaning
import features as features

if __name__ == "__main__":

    df = pd.read_json("data/raw/data.json")
    X_train, X_test = train_test_split(df,test_size = 0.2, random_state = 1)
    dc = DataCleaning(X_train)
    X, y = dc.prepare_data()
    X = features.minimal_df(X,['country','description'])



    # dc.create_y()
    #
    # dc.convert_dates()

    # dc.save_clean_json("data/processed/clean_data.json")
