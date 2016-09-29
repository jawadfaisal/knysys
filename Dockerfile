FROM ubuntu

RUN apt-get update && apt-get -y install python-dev pkg-config libfreetype6-dev openjdk-8-jre-headless libxml2-dev libboost-dev libboost-program-options-dev libboost-python-dev git build-essential libatlas-base-dev nano python-pip

ADD / /

RUN pip install -r /requirements.txt

ENTRYPOINT /bin/bash

