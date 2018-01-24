import pandas as pd
import tensorflow as tf
import collections
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.model_selection import GridSearchCV



def main():
    COLUMN_TYPES = collections.OrderedDict([
        ("crime", float),  # per capita crime rate by town 
        ("zn", float),     # proportion of residential land zoned for lots over 25,000 sq.ft. 
        ("indus", float),  # proportion of non-retail business acres per town 
        ("chas", int),     # Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
        ("nox", float),    # nitric oxides concentration (parts per 10 million) 
        ("rm", float),     # average number of rooms per dwelling 
        ("age", float),    # proportion of owner-occupied units built prior to 1940 
        ("dis", float),    # weighted distances to five Boston employment centres
        ("rad", int),      # index of accessibility to radial highways 
        ("tax", int),      # full-value property-tax rate per $10,000 
        ("ptratio", float), # pupil-teacher ratio by town ,
        ("b", float),      # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
        ("lstat", float),  # % lower status of the population 
        ("medv", float)    # Median value of owner-occupied homes in $1000's
    ])

    df = pd.read_csv('housingdata.csv', header=0, delimiter=';', names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES)
    
    seed = sum([ord(x) for x in "LIGHT FROM LIGHT"])
    np.random.seed(seed)

    # Training features, retain 0.9 for training, 0.1 for test (after cross validation and building of final model)
    df_sampled = df.sample(frac=1, random_state=seed)
    y_name = 'medv'
    mean = np.mean(df["medv"])
    df["medv] = df["medv"].apply(lambda x: (x - mean))
    # Features
    x_train, x_test = np.split(df_sampled, [int(0.9*len(df_sampled))])
    # Targets
    y_train = x_train.pop(y_name)
    y_test =  x_test.pop(y_name)

    # Grid search parameter space.
    #model = GradientBoostingRegressor(random_state=seed)
    #param_grid = { "loss"              : ['ls', 'huber', 'lad'],
    #               "n_estimators"      : [1,25,100,300,500],
    #               "max_depth"         : [1,3,6,9,12,15,18],
    #               "min_samples_split" : [2,4,6,8,10],
    #               "learning_rate"     : [0.01, 0.05, 0.1]}
    #grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=2, scoring='r2', verbose=1)
    #grid_search.fit(x_train, y_train)
    #print(grid_search.best_params_)

    # Construct final model.
    model = GradientBoostingRegressor(loss='huber', 
                                     max_depth=3, 
                                     min_samples_split=4, 
                                     n_estimators=300,
                                     learning_rate=0.1,
                                     random_state=seed)
    model.fit(x_train, y_train)
    y_pred = pd.Series(model.predict(x_test), index=y_test.index)
    print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred)) # Lower the better
    print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred)) # Lower the better
    print("Explained Variance Score: %.2f" % explained_variance_score(y_test, y_pred)) # Explained variance score: 1 is perfect
    print('R2 score: %.2f' % r2_score(y_test, y_pred)) # 1 is perfect

if __name__ == '__main__':
    main()