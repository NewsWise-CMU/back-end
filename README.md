# back-end

## To Run Locally with hotswap
- `pip install -r requirements.txt`
- `python -m uvicorn main:app --reload`

[See here for more details on installation](https://fastapi.tiangolo.com/tutorial/first-steps/)

## To Run with Docker
- `docker build -t myimage .`
- `docker run --name mycontainer -p 80:80 myimage`

[See here for more details on running this app using docker](https://fastapi.tiangolo.com/deployment/docker/)


## API Docs:
Once running you can find the docs here:
- [local run swagger docs](http://localhost:8080/docs)
- [local run redoc](http://localhost:8080/redoc)
- [docker run swagger](http://localhost:80/docs)
- [docker run redoc](http://localhost:80/redoc)
