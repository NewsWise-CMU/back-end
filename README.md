# back-end

## Notes
- To run this project you will need to specify an openai API key in the environment under the name "OPENAI_API_KEY"
    - alternatively, you can provide this in an .env file in the project root directory

## To Run Locally with hotswap
- `pip install -r requirements.txt`
- `python -m uvicorn main:app --reload`

[See here for more details on installation](https://fastapi.tiangolo.com/tutorial/first-steps/)

## To Run Locally with Docker
- `docker build -t fake_news_be -f $(PWD)/dev.Dockerfile .`
- `docker run -it -p 8000:8000 -v $(PWD)/app:/app/ fake_news_be`


## To Run with Docker
- `docker build -t fake_news_be .`
- `docker run -p 80:80 fake_news_be`

[See here for more details on running this app using docker](https://fastapi.tiangolo.com/deployment/docker/)


## API Docs:
Once running you can find the docs here:
- [local run swagger docs](http://localhost:8000/docs)
- [local run redoc](http://localhost:8000/redoc)
- [docker run swagger](http://localhost:8000/docs)
- [docker run redoc](http://localhost:8000/redoc)
