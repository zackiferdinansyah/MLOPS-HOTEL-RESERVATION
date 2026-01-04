import os

########################  DATA INGESTION  ########################

RAW_DIR = 'artifacts/raw'  # tentukan lokasi setelah data diextract dari gcp
RAW_FILE_PATH = os.path.join(RAW_DIR, 'raw.csv')
TRAIN_FILE_PATH = os.path.join(RAW_DIR, 'train.csv')
TEST_FILE_PATH = os.path.join(RAW_DIR, 'test.csv')

TRAIN_NAME_PATH = 'train.csv'
TEST_NAME_PATH = 'test.csv'

CONFIG_PATH = 'config/config.yaml'


########################  DATA PROCESSING  ########################

PROCESSED_DIR = 'artifacts/processed'  # buat folder processed didalam artifacts
PROCESSED_TRAIN_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_train.csv')  # simpan file processed_train.csv didalam folder processed
PROCESSED_TEST_DATA_PATH = os.path.join(PROCESSED_DIR, 'processed_test.csv')  # simpan file processed_test.csv didalam folder processed


########################  MODEL TRAINING  ########################

MODEL_OUTPUT_PATH = 'artifacts/models/lgbm_model.pkl'