import sys
import os
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataValidationConfig:
    def __init__(self):
        self.all_required_files = ["train","test","validation"]
        self.status_file = "artifacts/data_validation/status.txt"
        self.dataset_path = os.path.join("artifacts","data_ingestion","samsum_dataset")

class DataValidation:
    def __init__(self):
        self.data_validation = DataValidationConfig()

    def validate_all_file_exist(self)-> bool:
        try:
            validation_status = None
            all_files = os.listdir(os.path.join(self.data_validation.dataset_path))
            os.makedirs(os.path.dirname(self.data_validation.status_file), exist_ok=True)

            for file in all_files:
                if file not in self.data_validation.all_required_files:
                    validation_status = False
                    # with open(self.data_validation.status_file,"w") as f:
                    #     f.write(f"validation_status: {validation_status}")
                else:
                    validation_status = True
                    # with open(self.data_validation.status_file,"w") as f:
                    #     f.write(f"validation_status: {validation_status}")

            return validation_status

        except Exception as e:
            logging.info("Error in validate_all_file_exist")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    data_validation_obj = DataValidation()
    validation_status = data_validation_obj.validate_all_file_exist()
    print(validation_status)