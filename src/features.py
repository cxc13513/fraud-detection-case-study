import pandas as pd
import numpy as np

def impute_vals(df,col_name,val):
    '''
    Impute defined value val into col_name. Operates over one column at a time
    To run outside of class
    '''
    df[col_name] = df[col_name].fillna(val)
    return df

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
    df.drop('currency', axis=1, inplace=True)
    return df

def name_all_caps(df):
   '''new column of whether or not name of even is in all caps'''
   df['name_all_caps'] = map(lambda x: x==x.upper(), df.name)
   return df

def email_domains_to_ints(df):
    '''new colum that converts gmail, hotmail & yahoo domains to 1, others to 0s'''
    df['email_numeric'] = map(lambda x: ('hotmail.com' in x) or ('gmail.com' in x) or ('yahoo.com' in x), df.email_domain)
    df['email_numeric'] = df['email_numeric'].astype(int)
    df.drop('email_domain', axis=1, inplace=True)
    return df

def party_in_description(df):
   '''contains the fraud-likely word "party" in the description text'''
   df['party_in_description'] = map(lambda x: 'party' in x.lower(), df.description)
   return df

def pass_in_description(df):
   '''contains the fraud-likely word "pass" in the description text'''
   df['pass_description'] = map(lambda x: 'pass' in x.lower(), df.description)
   return df

def run_all(df,columns):
    copy = df.copy()
    copy = minimal_df(copy,columns)
    copy = currency_to_dollars(copy)
    copy = name_all_caps(copy)
    copy = email_domains_to_ints(copy)
    copy = party_in_description(copy)
    copy = pass_in_description(copy)
    return copy
