import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model_path = './fine-tuned-bert'
tokenizer = AutoTokenizer.from_pretrained(model_path + '-tokenizer')
model = AutoModelForSequenceClassification.from_pretrained(model_path + '-model')

trainer = Trainer(model = model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) 

def get_predicted_arg_score(text):
    df = pd.DataFrame({'text':[text]})
    dataset = Dataset.from_pandas(df,preserve_index=False) 
    tokenized_datasets = dataset.map(tokenize_function)
    raw_pred, _, _ = trainer.predict(tokenized_datasets)
    return(raw_pred[0][0])
