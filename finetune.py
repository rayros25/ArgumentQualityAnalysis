import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import mean_squared_error

# Load training data
df_train = pd.read_csv('./data/train_30k.csv')
df_train = df_train[['text', 'label']]

# Load testing data
df_test = pd.read_csv('./data/test_30k.csv')
df_test = df_test[['text', 'label']]

train_dataset = Dataset.from_pandas(df_train, preserve_index=False)
eval_dataset  = Dataset.from_pandas(df_test, preserve_index=False)


# Tokenize the data
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# Uncomment below to fine-tune the already existing fine-tuned model:
# tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-bert-tokenizer')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding    = 'max_length', 
                                       truncation = True, 
                                       max_length = 128)
train_dataset = train_dataset.map(tokenize_function, batched = True)
eval_dataset = eval_dataset.map(tokenize_function, batched = True)


# Load pre-trained BERT model and modify for regression
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased',
                                                           num_labels = 1)
# Uncomment below to fine-tune the already existing fine-tuned model:
# model = AutoModelForSequenceClassification.from_pretrained('./fine-tined-bert-model',
#                                                            num_labels = 1)


# Prepare the trainer, then train the model
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return {'mse': mean_squared_error(labels, predictions)}

training_args = TrainingArguments(
    output_dir                  = './results',
    num_train_epochs            = 3,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    warmup_steps                = 500,
    weight_decay                = 0.01,
    logging_dir                 = './logs',
    logging_steps               = 10,
    evaluation_strategy         = 'epoch'
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = eval_dataset,
    compute_metrics = compute_metrics
)

trainer.train()


# Save model and tokenizer
model.save_pretrained('./fine-tuned-bert-model')
tokenizer.save_pretrained('./fine-tuned-bert-tokenizer')
