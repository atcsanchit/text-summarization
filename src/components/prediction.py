import sys
from dataclasses import dataclass

from transformers import AutoTokenizer
from transformers import pipeline

from src.logger import logging
from src.exception import CustomException


@dataclass
class PredictionConfig:
    tokenizer_path = "artifacts/model_trainer/tokenizer"
    model_path = "artifacts/model_trainer/pegasus-samsum-model"

class Prediction:
    def __init__(self):
        self.prediction = PredictionConfig()
    
    def predict(self, text):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.prediction.tokenizer_path)
            gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}

            pipe = pipeline("summarization", model=self.prediction.model_path,tokenizer=tokenizer)

            print("Dialogue:")
            print(text)

            output = pipe(text, **gen_kwargs)[0]["summary_text"]
            print("\nModel Summary:")
            print(output)

            return output

        except Exception as e:
            logging.info("Error in predict")
            raise CustomException(e,sys)
        