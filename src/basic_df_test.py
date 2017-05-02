import numpy as np
import pandas as pd



def create_basic_df_test_pipeline(path):
    '''
    create fraud binary variable, keep numeric/float variables for testing
    INPUT: path to dataset
    OUTPUT: shortened pandas dataframe
    '''
    df = pd.read_json(path)
    # create new binary fraud variable based on certain conditions
    df['fraud'] = np.where((df['acct_type'] == 'fraudster_event')
                           | (df['acct_type'] == 'fraudster'), 1, 0)
    # check to make sure new fraud variable created okay
    df.fraud.value_counts(dropna=False)
    # keep only numeric columns of df
    df_num = df.select_dtypes(include=[np.float, np.number])
    return df_num
