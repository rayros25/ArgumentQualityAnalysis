# taken mostly from OpenAI API Docs

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
  organization = os.environ.get("ORG_ID"),
  project = os.environ.get("PROJ_ID"),
  api_key = os.environ.get("API_KEY"),
)

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

print(response.choices[0].message.content)
