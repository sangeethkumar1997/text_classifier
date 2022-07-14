FROM docker.io/python:3.8-slim
RUN apt-get update 
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENTRYPOINT ["./predict.py"]