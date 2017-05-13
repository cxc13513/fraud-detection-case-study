import pandas as pd
import numpy as np


def minimal_df(df,col_names, to_drop = False):
    '''
    input: list of col_names
    Returns a dataframe with only col_names leff, or drops col_names from the datafram if to_drop == True
    To run outside of class
    '''
    if to_drop:
        return df.drop(col_names,axis=1)
    return df[col_names]

def dummify(df, cols, constant_and_drop=False):
    '''
        Given a dataframe, for all the columns which are not numericly typed already,
        create dummies. This will NOT remove one of the dummies which is required for
        logistic regression (in such case, constant_and_drop should be True).
        To run outside of class
    '''
    df = pd.get_dummies(df, columns=cols, drop_first=constant_and_drop)
    if constant_and_drop:
        const = np.full(len(df), 1)
        df['constant'] = const
    return df

def currency_to_dollars(df):
    '''new column of booleans if currency == US Dollar'''
    df['currency_dollars'] = map(lambda x: 'USD' in x, df.currency)
    df = df.drop('currency', axis=1)
    return df

def name_all_caps(df):
   '''new column of whether or not name of even is in all caps'''
   df['name_all_caps'] = map(lambda x: x==x.upper(), df.name)
   df = df.drop('name', axis=1)
   return df

def email_domains_to_ints(df):
    '''new colum that converts gmail, hotmail & yahoo domains to 1, others to 0s'''
    df['email_numeric'] = map(lambda x: ('hotmail.com' in x) or ('gmail.com' in x) or ('yahoo.com' in x), df.email_domain)
    df['email_numeric'] = df['email_numeric'].astype(int)
    df = df.drop('email_domain', axis=1)
    return df

def party_in_description(df):
   '''contains the fraud-likely word "party" in the description text'''
   df['party_in_description'] = map(lambda x: 'party' in x.lower(), df.description)
   return df

def pass_in_description(df):
   '''contains the fraud-likely word "pass" in the description text'''
   df['pass_description'] = map(lambda x: 'pass' in x.lower(), df.description)
   df = df.drop('description', axis=1)
   return df

def time_between_user_event_created(df):
  df['time_between_user_event_created'] = (df.event_created - df.user_created).dt.days
  df = df.drop(['user_created','event_created'], axis=1)
  return df

def run_all(df,logistic=False):
    columns = ['body_length','currency',
                    'email_domain','name','num_order','num_payouts','has_logo','country','user_age']
    # columns = ['body_length','currency',
    #                 'email_domain','name','num_order','num_payouts','has_logo','event_created','user_created','country','user_age']
    copy = df.copy()
    copy = minimal_df(copy,columns)
    copy = currency_to_dollars(copy)
    # import pdb; pdb.set_trace()
    copy = name_all_caps(copy)
    copy = email_domains_to_ints(copy)
    # copy = party_in_description(copy)
    # copy = pass_in_description(copy)
    # copy = time_between_user_event_created(copy)
    copy = dummify(copy, ['email_numeric','currency_dollars', 'name_all_caps','has_logo','country'],constant_and_drop=logistic)
    # copy = dummify(copy, ['email_numeric', 'pass_description', 'party_in_description', 'currency_dollars', 'name_all_caps','has_logo','country'])
    return copy
