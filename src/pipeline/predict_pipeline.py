import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        
        
        try:
            model_path='artifacts\model.pkl'
            preprocessor_path = 'artifacts\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
class CustomData:
    def __init__(self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch:str,
        test_preparation_course:str,
        reading_score:int,
        writing_score:int):
        # Add None/empty value handling
        self.gender = gender if gender else 'male'
        self.race_ethnicity = race_ethnicity if race_ethnicity else 'group A'
        self.parental_level_of_education = parental_level_of_education if parental_level_of_education else 'some college'
        self.lunch = lunch if lunch else 'standard'
        self.test_preparation_course = test_preparation_course if test_preparation_course else 'none'
        self.reading_score = reading_score if reading_score else 0
        self.writing_score = writing_score if writing_score else 0
        
    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }
            
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e,sys)