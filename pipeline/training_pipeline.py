from src.data_ingestion import *
from src.data_preprocessing import *
from src.model_training import *


if __name__ == '__main__':
    # 1. Data Ingestion

    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    # 2. Data Preprocessing

    preprocessing = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    preprocessing.process()

    # 3. Model Training
    model_training = ModelTraining(PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH)
    model_training.run()

    # set GOOGLE_APPLICATION_CREDENTIALS=C:\Users\zacky ferdiansyah\Downloads\melodic-park-442312-k0-ae3c5c0fbe79.json