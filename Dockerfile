FROM python:3.12-slim-bookworm

WORKDIR /python-server

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD [ "python3", "-m", "flask", "run"]