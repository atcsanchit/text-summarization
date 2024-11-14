import sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.components.model_evaluation import ModelEvaluation


@dataclass
class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
            logging.info("Initiating model evaluation pipeline")
            print("Initiating model evaluation pipeline")

            model_evaluation_obj = ModelEvaluation()
            model_evaluation_obj.initiate_evaluation()

            print("model evaluation pipeline has been successfully executed")
            print("*"*20)        

        except Exception as e:
            logging.info("Error in main method")
            raise CustomException(e,sys)

if __name__ == "__main__":
    pipeline_obj = ModelEvaluationPipeline()
    pipeline_obj.initiate_pipeline()