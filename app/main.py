from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
openai.api_key_path = None
openai.api_key = os.getenv("OPENAI_API_KEY")


nltk.download('stopwords')
bias_model_name = "mediabiasgroup/DA-RoBERTa-BABE"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_name)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_name)


class ArticleContent(BaseModel):
    text: str


class FakeNewsPrediction(BaseModel):
    message: str
    label: str
    prob: float


class BiasScorePrediction(BaseModel):
    message: str
    bias_score: float

class ExtractedClaim(BaseModel):
    claim: str
    truthfulness: str
    possible_source: str


app = FastAPI()

origins = [
    "http://localhost:3000",
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


@app.get("/")
async def root():
    return {"message": "Welcome to NewsWise"}

@app.post("/predict-fake-news") 
async def root(article_content: ArticleContent) -> FakeNewsPrediction:
    input_text = truncate_string_using_embeddings(article_content.text, "text-ada-001", 2044)
    ft_model = 'ada:ft-personal-2023-04-18-19-13-19'
    res = openai.Completion.create(model=ft_model, prompt=input_text + '\n\n###\n\n', max_tokens=1, temperature=0, logprobs=10)
    label = res['choices'][0]['text']
    prob = np.exp(res['choices'][0]['logprobs']['top_logprobs'][0][label])
    return FakeNewsPrediction(message="Success", label=label, prob=prob)

@app.post("/predict-bias-score")
async def root(article_content: ArticleContent) -> BiasScorePrediction:
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
    return BiasScorePrediction(message="Success", bias_score=avg_bias_score)

@app.post("/extract-claims")
async def root(article_content: ArticleContent) -> list[ExtractedClaim]:
    input_text = article_content.text
    example_response = """
1. President George W. Bush claimed that Saddam Hussein was on the verge of developing nuclear weapons and was hiding other weapons of mass destruction as a key reason for invading Iraq. 
    * Truthfulness: Possibly True
    * Source: Politifact: Iraq war: 10 years on, where are the key players now?

2. The real reason for the invasion of Iraq was a long-range plan for "regime change" in the Middle East. 
    * Truthfulness: Opinion/Speculation
    * Source: N/A

3. The lack of evidence connecting Saddam Hussein to the 9/11 attacks made it difficult to rally the American people to support a war against Baghdad. 
    * Truthfulness: Possibly Fake
    * Source: The New York Times: Bush's Case for War: A Realpolitik Primer

4. The U.S. leadership cadre's worldview demanded a mortally dangerous Iraq, and informants who were willing to tell the tale of pending atomic weapons lowered the threshold for proof. 
    * Truthfulness: Possibly Fake
    * Source: The New York Times: Bush's Case for War: A Realpolitik Primer
"""
    valid_sources = "[encylopedia, textbook, sources with first hand reporting, research oriented magazine, Associated Press, BBC, C-SPAN, The Bureau of Investigative Journalism, The Economist, NPR, ProPublica, Reuters, USA Today, The Wall Street Journal, FAIR, Politifact]"
    invalid_sources = "[Fox News, The Guardian, Twitter, The Washington Post, The article itself.]"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are NewsWise, a fact and claim detector responsible for extracting the most important facts and claims from a piece of text and fact checking them with the most credible sources. Your goal is to find as many relevent, false claims as possible."},
            {"role": "user", "content": "I will be providing an article and you will give me a numbered list of the top 10 or less most important claims and facts relevant to the argument of the following article that deserve to be verified. Focus on claims that are false. In this list, discuss the truthfulness of each claim and add at least one third party source for each one other than the article itself. Give a truthfulness score that is only one of [Possibly True, Possibly False, Unknown, Opinion/Speculation] If the claim is an opinion then specify it. If the truthfulness of the claim cannot be determined then specify that. You should be highly critical of claims before making a decision. If it is a tossup or hard to determine, the class will be unknown. If the article presents an opinion than the truthfulness can only be an Opinion."},
            {"role": "user", "content": f"The sources should be one of {valid_sources}. Most importantly, make sure that none of the sources are the article itself or in this list: {invalid_sources}. Here is an example of the output required: {example_response}"},
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
        truthfulness = claim_data[1][claim_data[0].find("* Truthfulness: ") + 21:]
        possible_source = claim_data[2][claim_data[0].find("* Source: ") + 15:]
        response_data.append(ExtractedClaim(claim=claim, truthfulness=truthfulness, possible_source=possible_source))
    
    return response_data

 