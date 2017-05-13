import json
import sched, time
from predict import Predict
from train2 import get_fitted_model
from train_prepare import PrepareTrain
from sklearn import ensemble
from sklearn.model_selection import cross_val_score


from db_connect import DBConnector
import requests

import cPickle as pickle
import pandas as pd
from flask import Flask
from flask import (request,
                   redirect,
                   url_for,
                   session,
                   jsonify,
                   render_template)

app = Flask(__name__, template_folder='../templates', static_folder="../static")

@app.route('/')
def index():
    # return "boo"
    return render_template('index.html')


@app.route('/score', methods=['GET','POST'])
def score():

    d = requests.get(url).json()
    X = pd.DataFrame.from_dict(d, orient='index').T
    X['acct_type'] = 'pr'
    # data = pd.read_json('data/raw/data.json')

    support = copy.append(X)

    y = predict.predict(support)
    # y = True

    db.save_to_db(X,y[-1][0])
    return render_template('/show_json.html', table=X.T.to_html())


@app.route('/dashboard')
def dashboard():

    df = db.read_frm_db()

    history = df[['name','fraud']]
    return render_template('/dashboard.html', table=history.to_html())


if __name__ == '__main__':
    url = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"

    db = DBConnector()

    # s = sched.scheduler(time.time, time.sleep)
    # s.enter(60, 1, load_data, (s,))
    # s.run()
    # with open('random_forest.pkl') as f:
    #     model = pickle.load(f)
    data = pd.read_json('data/raw/data.json')
    copy = data.copy()
    X_all, y_all = PrepareTrain(data, undersample=False).prepare_data()
    model = get_fitted_model(X_all, y_all)
    predict = Predict(model)
    # df = pd.read_json('data/raw/data.json')
    # X_all, y_all = PrepareTrain(df, undersample=False).prepare_data()
    # model = get_fitted_model(X_all, y_all)
    app.run(host='0.0.0.0', port=8080, debug=True)
