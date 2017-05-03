import json
import sched, time


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

app = Flask(__name__, template_folder='../templates')

@app.route('/')
def index():
    # return "boo"
    return render_template('index.html')

@app.route('/submit')
def submit():
    return render_template('submit.html')

@app.route('/score', methods=['GET','POST'])
def score():

    d = requests.get(url).json()
    X = pd.DataFrame.from_dict(d, orient='index').T
    # y = model.predict(X)
    y = True

    db.save_to_db(X,y)
    return render_template('/show_json.html', table=X.to_html())

@app.route('/predict', methods=['POST'])
def predict():

    X = request.form['body']
    y_label = model.predict([X])

    return render_template('/predict.html', X=X, y_label=y_label[0])

@app.route('/dashboard')
def dashboard():

    df = db.read_frm_db()

    history = df[['name','fraud']]
    return render_template('/dashboard.html', table=history.to_html())


def load_data(sc):

    d = requests.get(url).json()
    X = pd.DataFrame.from_dict(d, orient='index').T
    # y = model.predict(X)
    y = True

    db.save_to_db(X,y)

    s.enter(30, 1, load_data, (sc,))


if __name__ == '__main__':
    url = "http://galvanize-case-study-on-fraud.herokuapp.com/data_point"

    db = DBConnector()

    # s = sched.scheduler(time.time, time.sleep)
    # s.enter(60, 1, load_data, (s,))
    # s.run()

    app.run(host='0.0.0.0', port=8080, debug=True)