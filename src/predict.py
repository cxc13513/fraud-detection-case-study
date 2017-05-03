import pandas as pd
from data_clean import DataCleaning
from features import run_all
from model_pipeline import ModelPipeline
import cPickle as pickle
from train_prepare import PrepareTrain
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from train2 import get_fitted_model

class Predict(object):

    def __init__(self,model):
        # self.X = raw_data
        # # with open(model) as f:
        # #     self.model = pickle.load(f)
        self.model = model

    def clean_data(self,data):
        return DataCleaning(data).convert_dates()

    def featurize(self,data):
        X = run_all(data)
        return X

    def predict(self,raw_data):
        # X = self.clean_data(raw_data)
        X = self.featurize(raw_data)
        prediction = self.model.predict_proba(X)
        return prediction

if __name__=="__main__":
    df = pd.read_json('data/raw/data.json')
    X_all, y_all = PrepareTrain(df, undersample=False).prepare_data()
    model = get_fitted_model(X_all, y_all)
    predict = Predict(model)
    prediction = predict.predict(df)
