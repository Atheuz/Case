import pandas as pd
import collections
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib

class HousePriceRegressor(object):
    def __init__(self):
        self.df_sampled = None
        self.model = None

    def load_data(self):
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

        # Establish seed for consistency.
        #seed = sum([ord(x) for x in "LIGHT FROM LIGHT"])
        #np.random.seed(seed)

        # Sample the full dataset randomly, i.e for the purpose of shuffling it.
        self.y_name = 'medv'
        self.df_sampled = df.sample(frac=1, random_state=1)
        self.df_sampled[self.y_name] = self.df_sampled[self.y_name]

    def split_data(self):
        # Features
        self.x_train, self.x_valid = np.split(self.df_sampled, [int(0.9*len(self.df_sampled))])
        self.x_train, self.x_test = np.split(self.x_train, [int(0.9*len(self.x_train))])
        # Targets
        self.y_train = self.x_train.pop(self.y_name)
        self.y_valid = self.x_valid.pop(self.y_name)
        self.y_test  = self.x_test.pop(self.y_name)

    def make_model(self):
        params = {'n_estimators': 5000, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'huber'}
        self.model = Pipeline([('regression', GradientBoostingRegressor(**params))])

    def fit_model(self):
        # Validation run
        self.model.fit(self.x_train, self.y_train)
        y_pred = pd.Series(self.model.predict(self.x_valid), index=self.y_valid.index)
        print("Mean squared error: %.2f" % mean_squared_error(self.y_valid, y_pred)) # Lower the better
        print("Mean absolute error: %.2f" % mean_absolute_error(self.y_valid, y_pred)) # Lower the better
        print("Explained Variance Score: %.2f" % explained_variance_score(self.y_valid, y_pred)) # Explained variance score: 1 is perfect
        print('R2 score: %.2f' % r2_score(self.y_valid, y_pred)) # 1 is perfect

    def test_model(self):
        # Finally test on test set.
        y_test_pred = pd.Series(self.model.predict(self.x_test), index=self.y_test.index)
        print("Mean squared error: %.2f" % mean_squared_error(self.y_test, y_test_pred)) # Lower the better
        print("Mean absolute error: %.2f" % mean_absolute_error(self.y_test, y_test_pred)) # Lower the better
        print("Explained Variance Score: %.2f" % explained_variance_score(self.y_test, y_test_pred)) # Explained variance score: 1 is perfect
        print('R2 score: %.2f' % r2_score(self.y_test, y_test_pred)) # 1 is perfect

    def save_model(self):
        joblib.dump(self.model, 'model.pkl') 

def main():
    hpr = HousePriceRegressor()
    hpr.load_data()
    hpr.split_data()
    hpr.make_model()
    hpr.fit_model()
    hpr.test_model()
    hpr.save_model()

if __name__ == '__main__':
    main()