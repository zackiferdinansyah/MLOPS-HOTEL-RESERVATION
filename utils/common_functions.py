import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml
import sys

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'file is not in the given path')
        
        with open(file_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info('succesfully read the YAML file')
            return config
    
    except Exception as e:
        logger.error('error while reading YAML file')
        raise CustomException('failed to  read YAML file', sys)
    

def load_data(path, name_path=False):
    try:
        if name_path==False:
            logger.info('loading the data')
        else:
            logger.info(f'Loading Data from {name_path}')

        return pd.read_csv(path)
    
    except Exception as e:
        logger.error(f'Error loading the data {e}')
        raise CustomException('Failed to load the data', sys)
    