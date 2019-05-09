FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
LABEL authors="Sina Masoud-Ansari, Tabish Rashid"

ENV SC2PATH /pymarl/3rdparty/StarCraftII

RUN apt-get update && apt-get install -y \
	git

COPY requirements.txt .

# Install python3
RUN apt-get -y install python3 python3-pip
RUN pip3 install --upgrade pip setuptools
RUN pip3 install -r requirements.txt

WORKDIR /pymarl
