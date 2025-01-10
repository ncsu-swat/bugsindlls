FROM mongodb/mongodb-community-server:latest
USER root
RUN apt-get update
RUN apt-get install -y python3.10
COPY FreeFuzz FreeFuzz/
WORKDIR /FreeFuzz
RUN pip install -r requirements.txt