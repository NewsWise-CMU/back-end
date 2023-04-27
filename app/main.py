from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import openai
import numpy as np
import tiktoken
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from pydantic import BaseModel
import sentry_sdk
from datetime import datetime


load_dotenv()
openai.api_key_path = None
openai.api_key = os.getenv("OPENAI_API_KEY")

data_file = "article_data.csv"
open(data_file, "a+")

nltk.download('stopwords')
bias_model_name = "mediabiasgroup/DA-RoBERTa-BABE"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_name)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)

class ArticleContent(BaseModel):
    text: str

class ArticleClaimContent(ArticleContent):
    claim: str

class BaseResponse(BaseModel):
    message: str

class FakeNewsPrediction(BaseResponse):
    label: str
    prob: float

class BiasScorePrediction(BaseResponse):
    bias_score: float

class ExtractedClaim(BaseModel):
    claim: str
    bias: str

class ExtractedClaims(BaseResponse):
    claims: list[ExtractedClaim]

class ClaimReasoning(BaseResponse):
    article_summary : str
    reasoning: str

class ArticleData(BaseResponse):
    article_data: str

sentry_sdk.init(
    dsn="https://0e2124a24b8b41f3b7830a26cee5741e@o4505064428666880.ingest.sentry.io/4505072500998144",

    traces_sample_rate=1.0,
    traces_sampler=1.0
)

app = FastAPI()


origins = [
    "http://localhost:3000",
    "https://www.news-wise-ai.com",
    "https://news-wise-ai.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def truncate_string_using_embeddings(string: str, model_name: str, num_tokens: int) -> str:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(string)[:num_tokens]
    return encoding.decode(tokens)

def add_article_data(article_str: str):
    with open(data_file, "a+") as myfile:
        myfile.write(f"{datetime.now()},{len(article_str.split(' '))}\n")

@app.get("/")
async def root():
    return {"message": "Welcome to NewsWise"}

@app.get("/article-data")
async def root() -> ArticleData:
    with open(data_file, "r") as myfile:
        return ArticleData(message="Success", article_data=myfile.read())

@app.post("/predict-fake-news")
async def root(article_content: ArticleContent) -> FakeNewsPrediction:
    ft_model = 'ada:ft-personal-2023-04-25-06-45-05'
    try:
        add_article_data(article_content.text)
        input_text = truncate_string_using_embeddings(article_content.text, "text-ada-001", 2044)
        res = openai.Completion.create(model=ft_model, prompt=input_text + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=10)
            
        label = res['choices'][0]['text']
        prob = np.exp(res['choices'][0]['logprobs']['top_logprobs'][0][label])
        return FakeNewsPrediction(message="Success", label=label, prob=prob)
    except:
        raise HTTPException(status_code=500, detail="Error during inference")

@app.post("/predict-bias-score")
async def root(article_content: ArticleContent) -> BiasScorePrediction:
    input_text = article_content.text
    # Tokenize the input text
    try:
        add_article_data(input_text)
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
        return BiasScorePrediction(message="Success", bias_score=avg_bias_score)
    except:
        raise HTTPException(status_code=500, detail="Error during inference")

@app.post("/extract-claims")
async def root(article_content: ArticleContent) -> ExtractedClaims:
    input_text = article_content.text
    example_output = """1. The previous charge d'affaires, Scott Hamilton, left Cuba earlier this month after the evacuation of all non-essential staff and relatives.
- Bias: The article does not provide any information on why Hamilton left Cuba, which could suggest a lack of transparency in the evacuation process.
2. Gumbiner will arrive at a time of heightened tensions between the United States and Cuba.
- Bias: The article does not provide any information on why tensions have increased, which could suggest a lack of transparency in the diplomatic relationship between the two countries."""
    try:
        add_article_data(input_text)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are NewsWise, a fact and claim detector responsible for extracting the most important facts and claims from a piece of text and discussing the various biases and stereotypes. Your goal is to find as many relevent, biased claims as possible."},
                {"role": "user", "content": f"I will be providing an article and you will give me a numbered list of the top 10 most important claims relevant to the argument of the article. In this list, discuss in detail any biases and linguistic techniques used in each claim that need to be acknowledged before taking it to heart. The output should look like this: {example_output}"},
                {"role": "user", "content": f"Here is the article:"},
                {"role": "user", "content": input_text}
            ],
            temperature=0
        )

        response_data = []
        response_content = response["choices"][0]["message"]["content"]
        for claim_data in response_content.split("\n\n"):
            claim_data = claim_data.split("\n")
            claim = claim_data[0][claim_data[0].find(" ") + 1:]
            bias = claim_data[1][claim_data[0].find("- Bias: ") + 9:]
            response_data.append(ExtractedClaim(claim=claim, bias=bias))
    
        return ExtractedClaims(message="Success", claims=response_data)

    except:
        raise HTTPException(status_code=500, detail="Error during inference")