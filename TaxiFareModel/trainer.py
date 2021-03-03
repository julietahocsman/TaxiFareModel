# imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        pipe_distance = make_pipeline(DistanceTransformer(), StandardScaler())
        pipe_time = make_pipeline(TimeFeaturesEncoder('pickup_datetime'), OneHotEncoder(handle_unknown='ignore'))
        dist_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        time_cols = ['pickup_datetime']

        preprocessing_pipe = ColumnTransformer([('time', pipe_time, time_cols),
                                          ('distance', pipe_distance, dist_cols)])
        pipe = Pipeline(steps=[('preprocessing_pipe', preprocessing_pipe),
                                ('regressor', LinearRegression())])

        return pipe

    def run(self, X_train, y_train, pipeline):
        """set and train the pipeline"""
        pipeline.fit(X_train, y_train)

        return pipeline

    def evaluate(self, X_test, y_test, pipeline):
        """evaluates the pipeline on df_test and return the RMSE"""

        y_pred = pipeline.predict(X_test)
        compute_rmse = np.sqrt(((y_pred - y_test)**2).mean())

        return compute_rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    print('TODO')






















