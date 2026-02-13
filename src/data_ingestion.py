#here we make data ingestion pipeline
import pandas as pd
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import logging

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#loggging configration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_data(data_path:str)->pd.DataFrame:
    """
    This function is used to load the data from the given path
    """
    try:
        df=pd.read_csv(data_path)
        logger.debug("Data loaded successfully from %s",data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file %s",data_path)
        raise ValueError("Empty data file")
    except pd.errors.EmptyDataError as e:
        logger.error("Error parsing data file")
        raise ValueError("Error parsing data file")
    except FileNotFoundError:
        logger.error("File not found at %s",data_path)
        raise FileNotFoundError("File not found at %s",data_path)
def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """
    This function is used to preprocess the data
    """
    try:
        df.rename(columns={'Message':'text','Category':'target'},inplace=True)
        logger.debug("Data preprocessing completed")
        return df
    except KeyError as e:
        logger.error(f"Missing columns in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise ValueError("Error preprocessing data")
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """
    This function is used to save the data to the given path
    """
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'),index=False)
        logger.debug("Data saved successfully to %s",data_path)
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise ValueError("Error saving data")

def main():
    try:
        test_size=0.2
        data_path = r'C:\Users\LAPTOPS HUB\Desktop\MLOps\End-to-End-ML-Pipeline\experiments\email.csv'
        df=load_data(data_path)
        final_df=preprocess_data(df)
        train_data,test_data=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_data,test_data,data_path='./data')
        logger.debug("Data ingestion completed successfully")
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        raise ValueError("Error in data ingestion")
if __name__=='__main__':
    main()  

        