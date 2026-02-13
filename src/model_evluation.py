import pandas as pd
import numpy as np
import os
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_recall_curve,f1_score
import pickle
import yaml
from dvclive import Live

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#loggging configration
logger=logging.getLogger('model_evluation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evluation.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    """
    This function is used to load the params from the given path
    """
    try:
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug("Params loaded successfully from %s",params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found at %s",params_path)
        raise FileNotFoundError("File not found at %s",params_path)
    except yaml.YAMLError as e:
        logger.error("Error parsing params.yaml file")
        raise ValueError("Error parsing params.yaml file")
    except Exception as e:
        logger.error("Error loading params.yaml file")
        raise ValueError("Error loading params.yaml file")



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

import json

# ... earlier code ...

def evaluate_model(model,X_test:np.ndarray,y_test:np.ndarray)->dict:
    """
    This function is used to evaluate the model
    """
    try:
        y_pred=model.predict(X_test)
        accuracy=accuracy_score(y_test,y_pred)
        report=classification_report(y_test,y_pred)
        cm=confusion_matrix(y_test,y_pred)
        logger.debug("Model evaluation completed")
        return {"accuracy":float(accuracy),"report":report,"confusion_matrix":cm.tolist()}
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        raise ValueError("Error evaluating model")

def save_metrics(metrics:dict,metrics_path:str)->None:
    """
    This function is used to save the metrics as JSON
    """
    try:
        os.makedirs(os.path.dirname(metrics_path),exist_ok=True)
        with open(metrics_path,'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug("Metrics saved successfully to %s",metrics_path)
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise ValueError("Error saving metrics")

def main():
    try:
        params=load_params(params_path='params.yaml')
        test_data=load_data(params['data_preprocessing']['test_data_path'])
        X_test=test_data.iloc[:,:-1]
        y_test=test_data.iloc[:,-1]
        model=pickle.load(open('./models/model.pkl','rb'))
        metrics = evaluate_model(model, X_test, y_test)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("accuracy", metrics["accuracy"])
            # Logging report and CM might need specific formatting for dvclive or just log summary metrics
            live.log_param("model_type", "pickle")
            live.log_param("params", params)

        save_metrics(metrics, './models/metrics.json')







        save_metrics(metrics,'./models/metrics.json')
        logger.debug("Model evaluation completed successfully")
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise ValueError("Error in model evaluation")


if __name__=='__main__':
    main()  