from encoders import DistanceTransformer, TimeFeaturesEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion


class Trainer:

    def __init__(self, model):
        self.model = model
        self.pipeline = None

    def set_pipeline(self, model):
        
        self.dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
    ])
        
        self.time_pipe = Pipeline([
        ('time_enc',TimeFeaturesEncoder('pickup_datetime')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])
        
        self.preproc_pipe = FeatureUnion([
        ('distance_pipeline', self.dist_pipe),
        ('time_features_pipeline', self.time_pipe)
    ])
        
        self.pipeline = Pipeline([("preprocessing", self.preproc_pipe),
                                  ("model", model)])

    def run(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        score = self.pipeline.score(X_test, y_test)
        return y_pred, score