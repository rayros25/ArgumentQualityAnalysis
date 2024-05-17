# This file is meant to compare different tunings of the fine-tuned BERT model
# provided

import predict_arg_score as pred
import pandas as pd


df_test = pd.read_csv('./data/med_test_30k.csv')
df_test = df_test[['text', 'label']]


avg = 0

for i in df_test.index:
    # print("exp:", df_test['label'][i], " pred:", pred.get_predicted_arg_score(df_test['text'][i]))
    diff = df_test['label'][i] - pred.get_predicted_arg_score(df_test['text'][i])
    diff = abs(diff)
    avg = avg + diff
    print(diff)

print("Avg: ", avg)
print("num tests: ", len(df_test.index))
print("Avg diff: ", avg / len(df_test.index))
