# Argument Quality Analysis

This web app, when given an argumentative statement, is meant to provide a score between 0 to 1 assessing how strong/compelling that statement was.
It does so via a fine-tuned version of Google's BERT model.
The training set used was IBM's [Debater Arg Quality Rank 30k](https://research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument_Quality).
Afterwards, it provides feedback from ChatGPT 3.5 turbo on the strengths and weaknesses of the argument.


## Install

*NOTE:* This repo uses Git Large File Storage to store tensor data from the fine-tuned BERT model. [Learn more here.](https://git-lfs.com/)

0. Set up a `.env` file containing OpenAI group ID, project ID, and API key for interfacing with ChatGPT. An example file is provided at `.env.example`.

1. In a Python virtual environment, run the following command:

`pip install pandas flask transformers datasets scikit-learn evaluate accelerate openai python-dotenv`

2. To run the web app, run the following command:

`flask --app argscore run`

## File Structure

- `data/`: training data and smaller subsets of it
- `fine-tuned-bert-model/`: data from fine-tuned BERT model
- `fine-tuned-bert-tokenizer/`: tokenizer from fine-tuned BERT model
- `static/`: CSS for web app
- `templates/`: HTML templates for web app
- `tests/`: directory for tests and/or code prototypes
- `.env.example`: example for `.env` file, which should hold API keys
- `.gitattributes`: boilerplate for Git LFS
- `.gitignore`: .gitignore
- `README.md`: README
- `argscore.py`: main web app
- `finetune.py`: script for fine-tuning the BERT model
- `get_advice.py`: function for interfacing with ChatGPT
- `predict_arg_score.py`: function for interfacing with fine-tuned BERT model

## Misc.

### Fine-tuning

To fine-tune the model, edit and run the `finetune.py` script.
By default, this fine-tunes Google's pre-trained BERT model from scratch using the ~6,000 testing points and ~21,000 training points.
On an M1 MacBook Air, this took about 2 hours. Smaller training/fine-tuning sets are in the `data/` directory.

# License

[IBM Debater® - IBM-ArgQ-Rank-30kArgs is released under CC-BY-SA](https://creativecommons.org/licenses/by-sa/3.0/)
