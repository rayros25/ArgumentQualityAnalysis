import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('./data/small_train_30k.csv')

# df = df[['argument', 'WA']]
# df =df[['text', 'labels']]

# Assuming your CSV has columns 'text' for the input text and 'score' for the labels
# TODO: text -> argument, score -> WA
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# Convert to Huggingface Dataset
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
# TODO: Why is this giving me errors?


from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# Apply tokenizer
train_dataset = train_dataset.map(tokenize_function, batched=True)
eval_dataset = eval_dataset.map(tokenize_function, batched=True)

# Set the format to PyTorch tensors
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


from transformers import AutoModelForSequenceClassification

# Load pre-trained BERT model and modify for regression
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)


from transformers import TrainingArguments, Trainer

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy='epoch'
)

# Define evaluation metric
import numpy as np
from sklearn.metrics import mean_squared_error

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.squeeze(logits)
    return {'mse': mean_squared_error(labels, predictions)}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()
# Save model and tokenizer
model.save_pretrained('./fine-tuned-bert')
tokenizer.save_pretrained('./fine-tuned-bert')
