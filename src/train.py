import pandas as pd
from sklearn.model_selection import train_test_split
from train_prepare import PrepareTrain

def split_data(path):
    df = pd.read_json(path)
    train, test = train_test_split(df,test_size = 0.2, random_state=1)
    return train, test

if __name__=="__main__":

    train, test = split_data('data/raw/data.json')
    X_train, y_train = PrepareTrain(train,undersample=False).prepare_data()

    #next step - creating the model
