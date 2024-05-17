import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

model_path = './fine-tuned-bert'
tokenizer = AutoTokenizer.from_pretrained(model_path + '-tokenizer')
model = AutoModelForSequenceClassification.from_pretrained(model_path + '-model')

trainer = Trainer(model = model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True) 

def pipeline_prediction(text):
    df=pd.DataFrame({'text':[text]})
    dataset = Dataset.from_pandas(df,preserve_index=False) 
    tokenized_datasets = dataset.map(tokenize_function)
    raw_pred, _, _ = trainer.predict(tokenized_datasets)
    return(raw_pred[0][0])

check_these = [ "a ban would only bring problems in gender equality",
                "a majority of americans identify with a religion.",
                "a defendant has a right to defend himself" ]

for argu in check_these:
    print(argu, ": ", pipeline_prediction(argu))

print("APPROX EXPECTED: 0.72, 0.52, 0.68")
