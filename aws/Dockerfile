FROM rocker/tensorflow
EXPOSE 5000
# RUN apt-get update 
RUN mkdir /usr/src/app
WORKDIR /usr/src/app
COPY ./requirements.txt .
# RUN apt-get -y install python3 python3-pip
RUN pip install -r requirements.txt
ENV PYTHONUNBUFFERED 1
COPY . .
ENTRYPOINT ["python3", "ml_project.py"]
