import os
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self, config):
        self.config = config['data_ingestion']              # diambil dari config/config.yaml
        self.bucket_name = self.config['bucket_name']       # diambil dari config/config.yaml
        self.file_name = self.config['bucket_file_name']    # diambil dari config/config.yaml
        self.train_test_ratio = self.config['train_ratio']  # diambil dari config/config.yaml

        os.makedirs(RAW_DIR, exist_ok=True)  # diambil dari path_config.py dibagian data ingestion

        logger.info(f'data ingestion started with {self.bucket_name} and file is {self.file_name}')
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f'csv file is succesfully download to {RAW_FILE_PATH}')

        except Exception as e:
            logger.error('error while downloading the csv file')
            raise CustomException('failed to download csv file', e)
        
    def split_data(self):
        try:
            logger.info('starting the splitting process')
            data = pd.read_csv(RAW_FILE_PATH)  # membaca dari config/paths_config.py, karena setelah diextrack dari gcp akan masuk ke raw.csv
            train_data, test_data = train_test_split(data, test_size=1-self.train_test_ratio, random_state=42)
            
            train_data.to_csv(TRAIN_FILE_PATH)  # setelah displit data masih berbentuk dataframe, akan diubah ke format csv, dan letakkan di TRAIN_FILE_PATH / path yang sudah dibuat sebelumnya di paths_config.py
            test_data.to_csv(TEST_FILE_PATH)    # setelah displit data masih berbentuk dataframe, akan diubah ke format csv, dan letakkan di TRAIN_FILE_PATH / path yang sudah dibuat sebelumnya di paths_config.py

            logger.info(f'train data saved to {TRAIN_FILE_PATH}')
            logger.info(f'test data saved to {TEST_FILE_PATH}')

        except Exception as e:
            logger.error('error while splitting data')
            raise CustomException('failed to split data into training ang testing data', e)
    
    def run(self):

        try:
            logger.info('starting data ingestion process')

            self.download_csv_from_gcp()
            self.split_data()
            
            logger.info('data ingestion completed succesfully')
        
        except CustomException as ex:
            logger.error(f'custom exception : {str(ex)}')

        finally:
            logger.info('data ingestion completed')


if __name__ == '__main__':
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))  # diambil dari config/paths_config.py
    data_ingestion.run()
