# FROM continuumio/miniconda3
FROM ubuntu:latest

WORKDIR /app

RUN apt-get update && apt-get install gcc python3-dev python3.9 pip ffmpeg -y

COPY ./scripts /app/scripts
COPY ./src /app/src
COPY ./utils /app/utils

COPY ./README.md /app/README.md
COPY ./requirements.txt /app/requirements.txt
COPY ./pyproject.toml /app/pyproject.toml
COPY ./setup.py /app/setup.py
COPY ./Makefile /app/Makefile


RUN pip install --upgrade pip
RUN pip install moviepy
RUN pip install -e .


RUN mkdir /app/runs
RUN mkdir /app/videos
VOLUME ./runs:/app/runs
VOLUME ./videos:/app/videos

CMD bash ./scripts/run.sh
