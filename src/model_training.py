# Here we write code about model training and we run our experiment model SVM,Naive Bayes,Logistic Regression
import pandas as pd
import numpy as np
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#loggging configration
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_training.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

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

def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->SVC:
    """
    Train the SVM linear model .
    :param X_train: Training data features
    :param y_train: Training data labels
    :param params: Hyperparameters for the model
    :return: Trained SVM model
    """
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples %d",X_train.shape[0],y_train.shape[0])
        logger.debug('initializing SVM model with parameters %s',params)
        model=SVC(kernel='linear',C=params['C'],gamma=params['gamma'])
        model.fit(X_train,y_train)
        logger.debug("Model training completed")
        return model
    except KeyError as e:
        logger.error(f"Missing columns in the dataframe: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise ValueError("Error in model training")

        
def save_model(model,model_path:str)->None:
    """
    This function is used to save the model to the given path
    save the model to a file
    :param model: model to be saved
    :param model_path: path to save the model
    """
    try:
        os.makedirs(os.path.dirname(model_path),exist_ok=True)
        with open(model_path,'wb') as f:
            pickle.dump(model,f)
        logger.debug("Model saved successfully to %s",model_path)
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise ValueError("Error saving model")
def main():
    try:
        params={'C':1,'gamma':1}
        train_data=load_data('./data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:,:-1]
        y_train=train_data.iloc[:,-1]
        model=train_model(X_train,y_train,params)
        save_model(model,'./models/model.pkl')
        logger.debug("Model training completed successfully")
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise ValueError("Error in model training")

if __name__=='__main__':
    main()  