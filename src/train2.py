import pandas as pd
# from sklearn.model_selection import train_test_split
from train_prepare import PrepareTrain
from sklearn import ensemble
from sklearn.model_selection import cross_val_score


def get_fitted_model(X, y):
    X_all_training = X
    y_all_training = y
    randomforest = ensemble.RandomForestClassifier(max_depth=1,
                                                   max_features='auto',
                                                   n_estimators=150)
    randomforest.fit(X_all_training, y_all_training)
    f1scores = cross_val_score(randomforest,
                               X_all_training, y_all_training,
                               cv=5, scoring='f1_weighted')
    recallscores = cross_val_score(randomforest,
                                   X_all_training, y_all_training,
                                   cv=5, scoring='recall_weighted')
    precisionscores = cross_val_score(randomforest, X_all_training,
                                      y_all_training, cv=5,
                                      scoring='precision_weighted')
    print("f1:", f1scores, "recall:", recallscores, "precision:", precisionscores)
    return randomforest

df = pd.read_json('data/raw/data.json')
X_all, y_all = PrepareTrain(df, undersample=False).prepare_data()
final_model_fitted = get_fitted_model(X_all, y_all)





#
#
# def split_data(path):
#     df = pd.read_json(path)
#     train, test = train_test_split(df,test_size = 0.2, random_state=1)
#     return train, test
#
# if __name__=="__main__":
#
#     #train, test = split_data('data/raw/data.json')
#     df = pd.read_json('data/raw/data.json')
#     X_all, y_all = PrepareTrain(df,undersample=False).prepare_data()
#
#     final_model_fitted = ModelPipeline.predict(X_all, y_all)
#
#     #next step - creating the model
