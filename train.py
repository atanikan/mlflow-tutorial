# Wine Quality Sample
import sys
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse


import mlflow
import mlflow.sklearn


import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# mlflow.set_experiment("wine_quality_linear_regression_2") # set experiment name
# mlflow.set_tracking_uri("http://127.0.0.1:8080") # set tracking server information
# mlflow.autolog() # automatically logs metrics 

    
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train(in_alpha = None, in_l1_ratio = None):
    np.random.seed(40)
    #mlflow.set_tracking_uri("http://127.0.0.1:8080") # set tracking server information

    # Read the wine-quality csv file
    #     csv_url ='http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    #     try:
    #         data = pd.read_csv(csv_url, sep=';')
    #     except Exception as e:
    #         logger.exception(
    #             "Unable to download training & test CSV, check your internet connection. Error: %s", e)

    wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
    data = pd.read_csv(wine_path)    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # Set default values if no alpha and no l1_ratio is provided
    alpha = float(in_alpha) if in_alpha else 0.5
    l1_ratio = float(in_l1_ratio) if in_l1_ratio else 0.5
    
    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
        # #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # signature = infer_signature(train_x, lr.predict(test_x))
        # mlflow.sklearn.log_model(
        #     sk_model=lr,
        #     artifact_path="sklearn-model",
        #     registered_model_name="sk-learn-random-forest-reg-model",
        #     signature=signature
        # )

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    train(sys.argv[1],sys.argv[2])