import pandas as pd


def set_cat(df,cat_list,dfsrc=None):
    for cat in cat_list:
        if dfsrc is not None and cat in dfsrc.columns:
            df[cat] = dfsrc[cat].astype('category').cat.as_ordered()
    return df

def train_cat(df,cat_list):
    for cat in cat_list:
         df[cat] = df[cat].astype('category').cat.as_ordered()
    return df
    
def numericalize_cat(df,cat_list):
    '''See https://pbpython.com/categorical-encoding.html'''
    for cat in cat_list:
         df[cat] = df[cat].astype('category').cat.codes
    return df
        