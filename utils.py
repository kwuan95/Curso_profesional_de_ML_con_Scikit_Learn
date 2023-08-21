import pandas as pd
import numpy as np
import sklearn
import joblib

class Utils:

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def load_from_mysql(self, path):
        return pd.read_mysql(path)

    def features_target(self, data, drop_cols, y):
        X = data.drop(drop_cols, axis=1)
        y = data[y]
        return X,y
    
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')
