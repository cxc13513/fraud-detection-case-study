#basic EDA
import numpy as np
import pandas as pd
import json
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from build_model_adapted import TextClassifier as TC

fraud_df_raw=pd.read_json('data.json')
fraud_df_raevent_published.unique()w.cfrom sklearn.feature_extraction.text import TfidfVectorizer as Tf olumns
fraud_df_raw.groupby('payout_type').count()

print fraud_df_raw.description[0:500]

fraud_df_raw.acct_type.unique()
fraud_df_raw['email_domain'][fraud_df_raw['acct_type']=='fraudster'].count()
gottext_df = fraud_df_raw.copy()
text=[BeautifulSoup(x, "lxml").get_text().replace('\\n', '\\n ') for x in gottext_df['description']]
trues=[x in ['fraudster', 'fraudster_event', 'fraudster_att'] for x in gottext_df.acct_type]
gottext_df['fraud']=trues
gottext_df['text']=text
gottext_df['fraud'].count()
print gottext_df['text'][gottext_df['fraud'] == True][:500]


#train test split gottext fraud and text
y = gottext_df['fraud']
X = gottext_df['text']

X_train, X_test, y_train, y_test= train_test_split(X,y)

tc=TC()
tc.fit(X_train, y_train)
tc.score(X_test, y_test)
tc.confusion_matrix(X_test, y_test)
