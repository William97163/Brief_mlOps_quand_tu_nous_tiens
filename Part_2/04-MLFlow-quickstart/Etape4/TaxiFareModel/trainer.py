from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion


class Trainer:

    def __init__(self, model):
        self.model = model

    def set_pipeline(self):
        
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
        
        time_pipe = Pipeline([
        ('time_enc',TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
        
        preproc_pipe = FeatureUnion([
        ('distance_pipeline', dist_pipe),
        ('time_features_pipeline', time_pipe)
    ])
        
        pipeline = Pipeline([("preprocessing", preproc_pipe),
                                  ("model", self.model)])
        return pipeline
    
    def run(self, X_train, y_train, pipeline):
        self.pipe = pipeline
        
        search = self.pipe.fit(X_train, y_train)
        return search

    def evaluate(self, X_test, y_test, search):
        self.search = search
        
        y_pred = self.search.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

