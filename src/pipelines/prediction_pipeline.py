import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Cement_component_1,
                 Blast_Furnace_Slag_component_2,
                 Fly_Ash_component_3,
                 Water_component_4,
                 Superplasticizer_component_5,
                 Coarse_Aggregate_component_6,
                 Fine_Aggregate_component_7,
                 Age_day):
        
        self.Cement_component_1 = Cement_component_1
        self.Blast_Furnace_Slag_component_2 = Blast_Furnace_Slag_component_2
        self.Fly_Ash_component_3 = Fly_Ash_component_3
        self.Water_component_4 = Water_component_4
        self.Superplasticizer_component_5 = Superplasticizer_component_5
        self.Coarse_Aggregate_component_6 = Coarse_Aggregate_component_6
        self.Fine_Aggregate_component_7 = Fine_Aggregate_component_7
        self.Age_day = Age_day
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Cement (component 1)(kg in a m^3 mixture)': [self.Cement_component_1],
                'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': [self.Blast_Furnace_Slag_component_2],
                'Fly Ash (component 3)(kg in a m^3 mixture)': [self.Fly_Ash_component_3],
                'Water  (component 4)(kg in a m^3 mixture)': [self.Water_component_4],
                'Superplasticizer (component 5)(kg in a m^3 mixture)': [self.Superplasticizer_component_5],
                'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': [self.Coarse_Aggregate_component_6],
                'Fine Aggregate (component 7)(kg in a m^3 mixture)': [self.Fine_Aggregate_component_7],
                'Age (day)': [self.Age_day]
            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)
