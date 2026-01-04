import os
import pandas as pd
import numpy as np
import sys
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import load_data, read_yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):  # ambil jalur dari data train & test setelah diextract dari GCP (config/paths_config.py)
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        # os.makedirs(self.processed_dir, exist_ok=True) --> bisa juga hanya pakai 1 baris ini

    def preprocessed_data(self, df, file_name_path):
        try:
            logger.info(f'starting data processing step to {file_name_path}')

            logger.info('dropping the columns')

            df.drop(columns=['Unnamed: 0', 'Booking_ID'], inplace=True)
            df.drop_duplicates(inplace=True)

            cat_cols = self.config['data_processing']['categorical_columns']
            num_cols = self.config['data_processing']['numerical_columns']

            logger.info('applying label encoding')

            label_encoder = LabelEncoder()
            mappings = {}  # membuat penampungan yang bertipe dictionary yg berisi (key:value)

            for col in cat_cols:
                df[col] = label_encoder.fit_transform(df[col])

                # melakukan label encoding & langsung mengubahnya didalam dataframe itu sendiri
                mappings[col] = {label:code for label, code in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}

            logger.info('label mapping are : ')
            for col, mapping in mappings.items():
                logger.info(f'{col} : {mapping}')
            
            logger.info('doing skewness handling')

            skew_threshold = self.config['data_processing']['skewness_threshold']
            skewness = df[num_cols].apply(lambda x:x.skew())  # ini akan menghasilkan dataframe [nama fitur, value skew]

            for column in skewness[skewness > skew_threshold].index:  # ambil fitur yang memiliki value skew > 5
                df[column] = np.log1p(df[column])  # kemudian ubah menggunakan logaritma (agar range nya tdk jauh berbeda)

            return df
        
        except Exception as e:
            logger.error(f'error during preprocess step {e}')
            raise CustomException('error while preprocess data', sys)
    

    def balance_data(self, df):  # parameter df ini diambil dari return method sebelumnya
        try:
            logger.info('handling imbalance data')

            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            smote = SMOTE(random_state=42)
            X_resample, y_resample = smote.fit_resample(X, y)

            balance_df = pd.DataFrame(X_resample, columns=X.columns)
            balance_df['booking_status'] = y_resample

            logger.info('data balance succesfully')

            return balance_df
        
        except Exception as e:
            logger.info(f'error during balanced data step {e}')
            raise CustomException('error during while balancing data', sys)
    

    def select_feature(self, df):
        try:
            logger.info('starting feature selection step')

            X = df.drop(columns=['booking_status'])
            y = df['booking_status']

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            result_features_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': result_features_importance
                })
            
            # urutkan berdasarkan value importance terbesar ke terkecil
            top_feature_importance_df = feature_importance_df.sort_values(by=['Importance'], ascending=False)

            # ambil 10 teratas dari features importance
            num_features_to_select = self.config['data_processing']['no_of_features']  # ambil dari config/config.yaml yg berisi 10 fitur yg nantinya akan diambil

            # ambil 10 fitur teratas yg sudah diurutkan
            top_10_features = top_feature_importance_df['Feature'].head(num_features_to_select).values

            logger.info(f'features selected {top_10_features}')

            top_10_df = df[top_10_features.tolist() + ['booking_status']]

            logger.info('features selection completed succesfully')

            return top_10_df
        
        except Exception as e:
            logger.info(f'error during feature selection step {e}')
            raise CustomException('error while feature selection', sys)
        

    def save_data(self, df, file_path):
        try:
            logger.info('saving our data in processed folder')

            df.to_csv(file_path, index=False)

            logger.info(f'data saved succesfully to {file_path}')

        except Exception as e:
            logger.error(f'error during save data step {e}')
            raise CustomException('error while saving data', sys)
        

    def process(self):  # method yang menggabungkan semua tahapan preprocess data
        try:
            logger.info('loading data from raw directory')

            # load data train dan test
            train_df = load_data(self.train_path, TRAIN_FILE_PATH)   # ambil file yg berada di jalur artifacts/raw/train.csv
            test_df = load_data(self.test_path, TEST_FILE_PATH)     # ambil file yg berada di jalur artifacts/raw/test.csv

            train_df = self.preprocessed_data(train_df, TRAIN_NAME_PATH)  # setelah mengambil data train kemudian lakukan preprocessed
            test_df = self.preprocessed_data(test_df, TEST_NAME_PATH)    # setelah mengambil data test kemudian lakukan preprocessed

            train_df = self.balance_data(train_df)      # lakukan balancing data hanya pada data (train.csv) saja
            # test_df = self.balance_data(test_df)

            train_df = self.select_feature(train_df)  # terapkan fitur selection based on (feature importance by random forest)
            test_df = test_df[train_df.columns]  # kolom yg dipilih dari data train, juga diterapkan ke data test

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)  # simpan ke jalur yg sudah disediakan pada config/paths_config.py
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info('data processing completed succesfully')

        except Exception as e:
            logger.error(f'error during preprocessing pipeline {e}')
            raise CustomException('error while data preprocessing pipeline', sys)
        

if __name__ == '__main__':
    data_preprocessing = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    data_preprocessing.process()