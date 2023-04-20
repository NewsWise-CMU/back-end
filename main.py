from fastapi import FastAPI
import os
from dotenv import load_dotenv
import openai
import requests
import numpy as np
import tiktoken
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from pydantic import BaseModel


load_dotenv()
CB_API_KEY = os.getenv('CB_API_KEY')
openai.api_key_path = None
openai.api_key = os.getenv("OPENAI_API_KEY")


nltk.download('stopwords')
bias_model_name = "mediabiasgroup/DA-RoBERTa-BABE"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_name)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)


class ArticleContent(BaseModel):
    text: str


app = FastAPI()

def truncate_string_using_embeddings(string: str, model_name: str, num_tokens: int) -> str:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(string)[:num_tokens]
    return encoding.decode(tokens)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict-fake-news")
async def root(article_content: ArticleContent):
    input_text = truncate_string_using_embeddings(article_content.text, "text-ada-001", 2044)
    ft_model = 'ada:ft-personal-2023-04-18-19-13-19'
    res = openai.Completion.create(model=ft_model, prompt=input_text + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=10)
    label = res['choices'][0]['text']
    return {
        "message": "Success",
        "label": label,
        "prob": np.exp(res['choices'][0]['logprobs']['top_logprobs'][0][label])
    }

@app.post("/predict-bias-score")
async def root(article_content: ArticleContent):
    input_text = article_content.text
    # Tokenize the input text
    input_tokens = bias_tokenizer.tokenize(input_text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in input_tokens if token.lower() not in stop_words]
    filtered_text = bias_tokenizer.convert_tokens_to_string(filtered_tokens)

    # Truncate the input to a maximum length of 512 tokens
    max_length = 512
    filtered_chunks = [filtered_text[i:i+max_length] for i in range(0, len(filtered_text), max_length)]

    # Get the predicted bias score for each chunk and average them
    bias_scores = []
    for chunk in filtered_chunks:
        inputs = bias_tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            outputs = bias_model(**inputs)
            logits = outputs.logits
            bias_score = logits.softmax(dim=1)[0][1].item()
            bias_scores.append(bias_score)

    avg_bias_score = sum(bias_scores) / len(bias_scores)
    return {
        "message": "Success",
        "bias_score": avg_bias_score,
    }