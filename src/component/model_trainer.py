import sys
import os
from dataclasses import dataclass

from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
import torch

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    root_dir: str = "artifacts/model_trainer"
    data_path: str = r"artifacts/data_transformation/samsum_dataset"
    model_ckpt: str = "google/pegasus-cnn_dailymail"
    num_train_epochs: int = 1
    warmup_steps: int = 500
    per_device_train_batch_size: int = 1
    weight_decay: float = 0.01
    logging_steps: int = 10
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_steps: float = 1e6
    gradient_accumulation_steps: int = 16

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()

    def initiate_training(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.model_trainer.model_ckpt)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.model_trainer.model_ckpt).to(device)
            seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)
            
            #loading data 
            dataset_samsum_pt = load_from_disk(self.model_trainer.data_path)

            # trainer_args = TrainingArguments(
            #     output_dir=self.model_trainer.root_dir, num_train_epochs=self.model_trainer.num_train_epochs, warmup_steps=self.model_trainer.warmup_steps,
            #     per_device_train_batch_size=self.model_trainer.per_device_train_batch_size, per_device_eval_batch_size=self.model_trainer.per_device_train_batch_size,
            #     weight_decay=self.model_trainer.weight_decay, logging_steps=self.model_trainer.logging_steps,
            #     evaluation_strategy=self.model_trainer.evaluation_strategy, eval_steps=self.model_trainer.eval_steps, save_steps=1e6,
            #     gradient_accumulation_steps=self.model_trainer.gradient_accumulation_steps
            # ) 


            trainer_args = TrainingArguments(
                output_dir=self.model_trainer.root_dir, num_train_epochs=1, warmup_steps=500,
                per_device_train_batch_size=1, per_device_eval_batch_size=1,
                weight_decay=0.01, logging_steps=10,
                evaluation_strategy='steps', eval_steps=500, save_steps=1e6,
                gradient_accumulation_steps=4
            ) 

            trainer = Trainer(model=model_pegasus, args=trainer_args,
                    tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                    train_dataset=dataset_samsum_pt["train"], 
                    eval_dataset=dataset_samsum_pt["validation"])
            
            trainer.train()

            ## Save model
            model_pegasus.save_pretrained(os.path.join(self.model_trainer.root_dir,"pegasus-samsum-model"))
            ## Save tokenizer
            tokenizer.save_pretrained(os.path.join(self.model_trainer.root_dir,"tokenizer"))

        except Exception as e:
            logging.error("Error in initiate_training")
            raise CustomException(e, sys)

if __name__ == "__main__":
    model_trainer_obj = ModelTrainer()
    dataset = load_from_disk(model_trainer_obj.model_trainer.data_path)
    print("dataset loading...")
    print(dataset["validation"])

    model_trainer_obj.initiate_training()
