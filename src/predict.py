import pandas as pd
from data_clean import DataCleaning
from features import run_all
from model_pipeline import ModelPipeline
# import pickle

class Predict(object):

    def __init__(self,raw_data,model):
        self.X = raw_data
        # with open(model) as f:
        #     self.model = pickle.load(f)
        self.model = model

    def clean_data(self):
        self.X = DataCleaning(self.X).convert_dates()

    def featurize(self):
        X = run_all(self.df,logistic=self.logistic)
        return X

    def predict(self):
        self.clean_data()
        self.featurize()
        prediction = self.model.predict_proba(self.X)
        return prediction
