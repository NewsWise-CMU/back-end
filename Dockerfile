# 
FROM python:3.10

RUN apt update && \
    apt install -y \
    python3-openssl \
    libxmlsec1-openssl
# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
COPY ./.env /code/app/

# copy certificate
COPY fullchain.pem /code
COPY privkey.pem /code

ENTRYPOINT [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "443", \
    "--ssl-keyfile", "privkey.pem", \
    "--ssl-certfile", "fullchain.pem" \
]