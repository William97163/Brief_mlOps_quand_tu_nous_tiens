import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from TaxiFareModel.utils import haversine_vectorized

class DistanceTransformer(TransformerMixin, BaseEstimator):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
        CORRECTION: returns the whole dataframe (impossible to use both pipelines
        at once otherwise)
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        
    def fit(self, X, y=None): return self

    def transform(self, X, y=None):
    
        X_ = X.copy()
        distance = haversine_vectorized(X)
        X_["distance"] = distance
        return X_[["distance"]]
    

# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self,datetime="pickup_datetime"):
           self.datetime = datetime

    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        # A COMPLETER 
        X_ = X.copy()
        X_[self.datetime] = pd.to_datetime(X[self.datetime], format='%Y-%m-%d %H:%M:%S UTC')
    
        # extract hour, day of week, month, and year
        X_['hour'] = X_[self.datetime].dt.hour
        X_['dow'] = X_[self.datetime].dt.dayofweek
        X_['month'] = X_[self.datetime].dt.month
        X_['year'] = X_[self.datetime].dt.year
        
        return X_[['dow', 'hour', 'month', 'year']]
