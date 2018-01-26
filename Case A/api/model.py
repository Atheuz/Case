import pandas as pd
import numpy as np
from sklearn.externals import joblib
import json
import collections

def load_model():
    model = joblib.load('model.pkl')
    return model

def predict(params):
    # Define column names, types
    COLUMN_TYPES = collections.OrderedDict([
        ("crime", float), # Important
        ("zn", float),
        ("indus", float),
        ("chas", int),
        ("nox", float),
        ("rm", float), # Important! Without this accuracy drops
        ("age", float),
        ("dis", float), # Important! 
        ("rad", int),
        ("tax", int),
        ("ptratio", float), # Important!
        ("b", float), # Doesnt do much
        ("lstat", float), # Important
    ])
    # Read into pandas dataframe
    X = pd.DataFrame(columns=COLUMN_TYPES.keys())
    d = json.loads(params)
    X = X.append({"crime":     d["crime"],
               "zn":           d["zn"],
               "indus":        d["indus"],
               "chas":         d["chas"], 
               "nox":          d["nox"],
               "rm":           d["rm"],
               "age":          d["age"],
               "dis":          d["dis"],
               "rad":          d["rad"],
               "tax":          d["tax"],   
               "ptratio":      d["ptratio"],
               "b":            d["b"],
               "lstat":        d["lstat"]}, ignore_index=True)

    model = load_model()
    y = float(model.predict(X)[0])
    return y

if __name__ == '__main__':
    predict()