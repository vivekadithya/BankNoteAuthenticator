FROM python:3.9

COPY . /usr/app/

EXPOSE 5000

WORKDIR /usr/app/

RUN pip3 install -r requirements.txt

CMD python3 app.py