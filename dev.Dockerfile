FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

COPY ./.env /code/app/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN cd /app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--reload"]
