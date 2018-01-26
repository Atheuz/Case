import pandas as pd
import tensorflow as tf
import collections
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

def main():
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
        ("medv", float)
    ])
    # Read into pandas dataframe
    df = pd.read_csv('housingdata.csv', header=0, delimiter=';', names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES)

    # Establish seed for consistency.
    seed = sum([ord(x) for x in "LIGHT FROM LIGHT"])
    np.random.seed(seed)

    # Sample the full dataset randomly, i.e for the purpose of shuffling it. Also remove the mean from targets.
    y_name = 'medv'
    mean = np.mean(df["medv"])
    df_sampled = df.sample(frac=1, random_state=seed)
    df["medv"] = df["medv"].apply(lambda x: (x - mean))

    # Features
    x_train, x_valid = np.split(df_sampled, [int(0.9*len(df_sampled))])
    x_train, x_test = np.split(x_train, [int(0.9*len(x_train))])
    # Targets
    y_train = x_train.pop(y_name)
    y_valid = x_valid.pop(y_name)
    y_test  = x_test.pop(y_name)

    model = Pipeline([('regression', GradientBoostingRegressor(loss='huber', 
                                                               max_depth=3, 
                                                               min_samples_split=4, 
                                                               n_estimators=300,
                                                               learning_rate=0.1,
                                                               random_state=seed))
    ])

    # Validation run
    model.fit(x_train, y_train)
    y_valid = y_valid
    y_pred = pd.Series(model.predict(x_valid), index=y_valid.index)
    print("Mean squared error: %.2f" % mean_squared_error(y_valid, y_pred)) # Lower the better
    print("Mean absolute error: %.2f" % mean_absolute_error(y_valid, y_pred)) # Lower the better
    print("Explained Variance Score: %.2f" % explained_variance_score(y_valid, y_pred)) # Explained variance score: 1 is perfect
    print('R2 score: %.2f' % r2_score(y_valid, y_pred)) # 1 is perfect

    # Finally test on test set.
    y_test_pred = pd.Series(model.predict(x_test), index=y_test.index)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_test_pred)) # Lower the better
    print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_test_pred)) # Lower the better
    print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_test_pred)) # Explained variance score: 1 is perfect
    print('R2 score: %.2f' % r2_score(y_test, y_test_pred)) # 1 is perfect

    # Good enough! Model is finished, now it can be saved and hosted that can be queried through an API endpoint. Out of scope for this.

if __name__ == '__main__':
    main()