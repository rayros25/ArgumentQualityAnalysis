import os
from dotenv import load_dotenv
from openai import OpenAI

# Get keys and connect to OpenAI
load_dotenv()
client = OpenAI(
    organization = os.environ.get("ORG_ID"),
    project = os.environ.get("PROJ_ID"),
    api_key = os.environ.get("API_KEY"),
)

# get_advice() returns what ChatGPT has to say about the provided argument
def get_advice(argument, score):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant with years of experience in debate."},
            {"role": "user", "content": "On a scale from 0 to 1, the following argument received a score of " + str(score) + ": " + argument},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "This is only the main argument, not the entire work. Only consider the main argument, and do not consider expanding it. Do not consider counterarguments."},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "What are the strengths of this argument? What are the weaknesses?"}
        ]
    )
    return response.choices[0].message.content

# get_advice_f() is the formatted version of get_advice(). The last prompt can
# be edited so that the reply is in HTML formatting.
def get_advice_f(argument, score):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant with years of experience in debate."},
            {"role": "user", "content": "On a scale from 0 to 1, the following argument received a score of " + str(score) + ": " + argument},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "This is only the main argument, not the entire work. Only consider the main argument, and do not consider expanding it. Do not consider counterarguments. Do not consider a lack of examples."},
            {"role": "assistant", "content": "I understand."},
            {"role": "user", "content": "What are the strengths of this argument? What are the weaknesses? Answer concisely but without using bullet points."}
        ]
    )
    return response.choices[0].message.content


# Simple test cases
def get_advice_test_millitary():
    print(get_advice("a military company ban would only be closing opportunities", 0.344499592))

def get_advice_test_religion():
    print(get_advice("a person's beliefs are bigger than what they are forced to do during their school day.  taking a brief time to allow students to engage in their religion should be encouraged.", 0.85635224))
