FROM python:3.10-bullseye

RUN mkdir /app

COPY *.py /app/
COPY requirements.txt /app/
COPY /etc/secrets/.env /app/.env

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 7860

CMD ["python3", "bot.py"]
