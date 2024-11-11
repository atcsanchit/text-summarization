import sys
import os
from dataclasses import dataclass

from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from src.logger import logging
from src.exception import CustomException


@dataclass
class DataTransformationConfig:
    def __init__(self):
        pass
        self.root_dir = "artifacts/data_transformation"
        self.data_path = os.path.abspath("artifacts/data_ingestion/samsum_dataset")
        self.tokenizer_name = "google/pegasus-cnn_dailymail"

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_transformation.tokenizer_name)

    def convert_examples_to_features(self,example_batch):
        try:
            input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )
            
            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(example_batch['summary'], max_length = 128, truncation = True )
                
            return {
                'input_ids' : input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }

        except Exception as e:
            logging.info("Error in convert_examples_to_features")
            raise CustomException(e,sys)

    def convert_dataset(self):
        try:
            dataset_samsum = load_from_disk(self.data_transformation.data_path)
            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
            dataset_samsum_pt.save_to_disk(os.path.join(self.data_transformation.root_dir,"samsum_dataset"))

        except Exception as e:
            logging.info("Error in convert_dataset")
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    data_transformation_obj = DataTransformation()
    # print("***********************")
    # # print("Loading dataset from:", data_transformation_obj.data_path)

    data_transformation_obj.convert_dataset()