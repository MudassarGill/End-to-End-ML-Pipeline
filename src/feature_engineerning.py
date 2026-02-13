import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#loggging configration
logger=logging.getLogger('feature_engineerning')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'feature_engineerning.log')
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
        df.fillna('',inplace=True)
        logger.debug("Data loaded and missing values filled successfully from %s",data_path)
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

def apply_tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame)->pd.DataFrame:
    """
    This function is used to apply tfidf vectorization to the data
    """
    try:
        tfidf=TfidfVectorizer()
        X_train=train_data['text']
        X_test=test_data['text']
        y_train=train_data['target']
        y_test=test_data['target']

        X_train_tfidf=tfidf.fit_transform(X_train)
        X_test_tfidf=tfidf.transform(X_test)

        train_df=pd.DataFrame(X_train_tfidf.toarray())
        train_df['target']=y_train

        test_df=pd.DataFrame(X_test_tfidf.toarray())
        test_df['target']=y_test

        logger.debug("TFIDF Applied successfully")
        return train_df,test_df
    except KeyError as e:
        logger.error(f"Missing columns in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise ValueError("Error preprocessing data")

def save_data(df:pd.DataFrame,data_path:str)->None:
    """
    This function is used to save the data to the given path
    """
    try:
        os.makedirs(os.path.dirname(data_path),exist_ok=True)
        df.to_csv(data_path,index=False)
        logger.debug("Data saved successfully to %s",data_path)
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise ValueError("Error saving data")

def main():
    try:
        max_features=50
        train_data=load_data('./data/interim/train_processed_data.csv')
        test_data=load_data('./data/interim/test_processed_data.csv')
        train_df,test_df=apply_tfidf(train_data,test_data)
        save_data(train_df,os.path.join('./data/processed/train_tfidf.csv'))
        save_data(test_df,os.path.join('./data/processed/test_tfidf.csv'))
        logger.debug("feature engineering completed successfully")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise ValueError("Error in feature engineering")
if __name__=='__main__':
    main()  