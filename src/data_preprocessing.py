import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
import os

#logggig configration
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

#loggging configration
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.addHandler(console_handler)

logger.debug("Downloading NLTK resources...")
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
logger.debug("NLTK resources downloaded successfully")

# Initialize NLTK objects
ps = PorterStemmer()
STOP_WORDS = set(stopwords.words('english'))
PUNCTUATION = set(string.punctuation)

def transform_text(text):
    """
    This function is used to transform the text:
    1. Lowercase
    2. Tokenize
    3. Remove stop words and punctuation
    4. Stemming
    """
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    
    transformed_tokens = [
        ps.stem(word) 
        for word in tokens 
        if word not in STOP_WORDS and word not in PUNCTUATION and word.isalnum()
    ]
    
    return ' '.join(transformed_tokens)

def preprocess_df(df,text_column='text',target_column='target'):
    """
    This function is used to preprocess the data
    """
    try:
        logger.debug("Starting Preprocessing the dataFrame")
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        
        logger.debug("Target column  encoded completed")

        df =df.drop_duplicates(keep='first')
        logger.debug("Duplicate values removed")

        df.loc[:,text_column]=df.loc[:,text_column].apply(transform_text)
        logger.debug("Text column transformed")
        return df
    except KeyError as e:
        logger.error("Error columns not found in Preprocessing the dataFrame")
        raise e
    except Exception as e:
        logger.error("Error in Preprocessing the dataFrame")
        raise e
    
def main():
    """
    This function is used to preprocess the data
    """
    try:
        train_data_path=r'data/raw/train_data.csv'
        test_data_path=r'data/raw/test_data.csv'
        
        # Load the data
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data_path)
        logger.debug("Loading the data successfully")
        

        #transform the data
        train_processed_data=preprocess_df(train_df)
        test_processed_data=preprocess_df(test_df)

        #store the data inside data/processed folder
        data_path=os.path.join("./data",'interim')
        os.makedirs(data_path,exist_ok=True)
        train_processed_data.to_csv(os.path.join(data_path,'train_processed_data.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed_data.csv'),index=False)
        logger.debug("Data preprocessing completed successfully and saved to %s",data_path)
    except Exception as e:
        logger.error("Error in data preprocessing: %s", str(e))
        raise e
    
if __name__=='__main__':
    main()



