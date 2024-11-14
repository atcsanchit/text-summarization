import sys
import os
from dataclasses import dataclass
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import evaluate 
import torch
import pandas as pd
from tqdm import tqdm
import os

from src.logger import logging
from src.exception import CustomException


@dataclass

class ModelEvaluationConfig:
    root_dir = "artifacts/model_evaluation"
    data_path = "artifacts/data_transformation/samsum_dataset"
    model_path = "artifacts/model_trainer/pegasus-samsum-model"
    tokenizer_path = "artifacts/model_trainer/tokenizer"
    metric_file_name = "artifacts/model_evaluation"

class ModelEvaluation:
    def __init__(self):
        self.model_evaluation = ModelEvaluationConfig()

    def generate_batch_sized_chunks(self,list_of_elements,batch_size):
        try:
            for i in range(0, len(list_of_elements), batch_size):
                yield list_of_elements[i:i+batch_size]

        except Exception as e:
            logging.info("Error in generate_batch_sized_chunks")
            raise CustomException(e,sys)
    
    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, batch_size = 16, device="cuda" if torch.cuda.is_available() else "cpu",
                                   column_text="article",column_summary="highlights"):
        try:
            article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
            target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total = len(article_batches)):
                
                inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device), 
                    length_penalty=0.8, num_beams=8, max_length=128
                )
                decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True,clean_up_tokenization_spaces=True) for s in summaries]     
                decoded_summaries = [d.replace(""," ") for d in decoded_summaries]
                metric.add_batch(predictions=decoded_summaries, references=target_batch)

            score = metric.compute()
            return score

        except Exception as e:
            logging.info("Error in calculate_metric_on_test_ds")
            raise CustomException(e,sys)
    
    def initiate_evaluation(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.model_evaluation.tokenizer_path)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.model_evaluation.model_path).to(device)
            
            dataset_samsum_pt = load_from_disk(self.model_evaluation.data_path)

            # rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

            rouge_metric = evaluate.load("rouge")

            score = self.calculate_metric_on_test_ds(

            dataset_samsum_pt['test'][0:10], rouge_metric, model_pegasus, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
                )

            df = pd.DataFrame(score, index=['pegasus'])

            directory = os.path.dirname(self.model_evaluation.metric_file_name)
            if not os.path.exists(directory):
                os.makedirs(directory)

            df.to_csv(self.model_evaluation.metric_file_name + "/metric.csv", index=False)
            print(f"Metrics saved to '{self.model_evaluation.metric_file_name}'")
            return score

        except Exception as e:
            logging.info("Error in initiate_evaluation")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    model_evaluation_obj = ModelEvaluation()
    score = model_evaluation_obj.initiate_evaluation()    