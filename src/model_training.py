import os
import pandas as pd
import joblib
import sys
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from config.model_params import *
from utils.common_functions import read_yaml, load_data
from scipy.stats import randint

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:

    def __init__(self, train_path, test_path, model_output_path):  # akan mengambil jalur processed/processed_train.csv & processed_test.csv
        self.train_path = train_path                             # dan model_output_path akan mengambil jalur tmpt menyimpan model
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    
    def load_and_split_data(self):
        try:
            logger.info(f'loading data from {self.train_path}')
            train_df = load_data(self.train_path)

            logger.info(f'loading data from {self.test_path}')
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=['booking_status'])
            y_train = train_df['booking_status']

            X_test = test_df.drop(columns=['booking_status'])
            y_test = test_df['booking_status']

            logger.info('data splitting successfully for model training')

            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            logger.error(f'error while loading data {e}')
            raise CustomException('failed to load data', sys)
    

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info('initializing our model')

            lgbm_model = LGBMClassifier(random_state=self.random_search_params['random_state'])

            logger.info('starting our hyperparameter tuning')

            random_search = RandomizedSearchCV(
                estimator=lgbm_model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params['n_iter'],
                scoring=self.random_search_params['scoring'],
                n_jobs=self.random_search_params['n_jobs'],
                cv=self.random_search_params['cv'],
                verbose=self.random_search_params['verbose'],
                random_state=self.random_search_params['random_state']
            )

            logger.info('starting our hyperparameter tuning...')

            random_search.fit(X_train, y_train)

            logger.info('hyperparameter tuning completed')

            best_params = random_search.best_params_
            best_lgbm_model = random_search.best_estimator_

            logger.info(f'best parameters are : {best_params}')

            return best_lgbm_model
        
        except Exception as e:
            logger.error(f'error while training model {e}')
            raise CustomException('failed to train model', sys)
        

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info('evaluating our model')

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f'accuracy score : {accuracy}')
            logger.info(f'precision score : {precision}')
            logger.info(f'recall score : {recall}')
            logger.info(f'f1 score : {f1}')

            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1 score': f1
            }

        except Exception as e:
            logger.error(f'error while evaluating model {e}')
            raise CustomException('failed to evaluate model', sys)
        

    def save_model(self, model):
        try:
            # if not os.path.exists(self.model_output_path):
            #     os.makedirs(self.model_output_path)  # akan dibuat directory jika file model belum disimpan / tdk rekomended

            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)  #--> buat directory/jika sudah ada biarkan saja / rekomended

            logger.info('saving the model')

            joblib.dump(model, self.model_output_path)  # model akan disimpan ke dalam directory : artifacts/models/lgbm_model.pkl

            logger.info(f'model saved to {self.model_output_path}')

        except Exception as e:
            logger.error(f'error while saving model {e}')
            raise CustomException('failed to save model', sys)
        
    
    def run(self):
        try:
            with mlflow.start_run():
                logger.info('starting our model training pipeline')

                logger.info('starting our mlflow experimentation')

                logger.info('logging the training and testing dataset to mlflow')
                mlflow.log_artifact(self.train_path, artifact_path='datasets')
                mlflow.log_artifact(self.test_path, artifact_path='datasets')

                X_train, y_train, X_test, y_test = self.load_and_split_data()
                best_lgbm_model = self.train_lgbm(X_train, y_train)
                metrics = self.evaluate_model(best_lgbm_model, X_test, y_test)
                self.save_model(best_lgbm_model)

                logger.info('logging the model into mlflow')
                mlflow.log_artifact(self.model_output_path)

                logger.info('logging params and metrics to mlflow')
                mlflow.log_params(best_lgbm_model.get_params())
                mlflow.log_metrics(metrics)

                logger.info('model training successfully completed')

        except Exception as e:
            logger.error(f'error in model training pipeline {e}')
            raise CustomException('failed during model training pipeline', sys)
        

if __name__ == '__main__':
    model_training = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    model_training.run()