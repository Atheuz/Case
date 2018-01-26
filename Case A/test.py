import pandas as pd
import numpy as np
import collections
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score

# Read dataset into X and Y
# Define column names, types
COLUMN_TYPES = collections.OrderedDict([
    ("crime", float), 
    ("zn", float),
    ("indus", float),
    ("chas", int),
    ("nox", float),
    ("rm", float), 
    ("age", float),
    ("dis", float), 
    ("rad", int),
    ("tax", int),
    ("ptratio", float),
    ("b", float), 
    ("lstat", float),
    ("medv", float)
])
# Read into pandas dataframe
df = pd.read_csv('housingdata.csv', header=0, delimiter=';', names=COLUMN_TYPES.keys(), dtype=COLUMN_TYPES)
df = df.sample(frac=1, random_state=1)
y_name = 'medv'
x_train, x_valid = np.split(df, [int(0.9*len(df))])
y_train = x_train.pop(y_name)
y_valid = x_valid.pop(y_name)

# Define the neural network
from keras.models import Sequential
from keras.layers import Dense

def build_nn():
    model = Sequential()
    model.add(Dense(20, input_dim=13, init='normal', activation='relu'))
    # No activation needed in output layer (because regression)
    model.add(Dense(1, init='normal'))

    # Compile Model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Evaluate model (kFold cross validation)
from keras.wrappers.scikit_learn import KerasRegressor

# sklearn imports:
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Before feeding the i/p into neural-network, standardise the dataset because all input variables vary in their scales
estimators = []
#estimators.append(('standardise', StandardScaler()))
estimators.append(('multiLayerPerceptron', KerasRegressor(build_fn=build_nn, nb_epoch=1000, batch_size=5, verbose=0)))

pipeline = Pipeline(estimators)

#kfold = KFold(n=len(x_train), n_folds=10)
#results = cross_val_score(pipeline, x_train, y_train, cv=kfold)

pipeline.fit(x_train, y_train)
y_pred = pd.Series(pipeline.predict(x_valid), index=y_valid.index)
print(y_pred)
print(y_valid)
print("Mean squared error: %.2f" % mean_squared_error(y_valid, y_pred)) # Lower the better
print("Mean absolute error: %.2f" % mean_absolute_error(y_valid, y_pred)) # Lower the better
print("Explained Variance Score: %.2f" % explained_variance_score(y_valid, y_pred)) # Explained variance score: 1 is perfect
print('R2 score: %.2f' % r2_score(y_valid, y_pred)) # 1 is perfect