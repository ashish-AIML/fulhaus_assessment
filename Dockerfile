FROM python:3.8-slim-buster

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 6000

CMD ["python3", "app.py"]
