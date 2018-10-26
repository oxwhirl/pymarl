FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04
MAINTAINER Tabish Rashid

# CUDA includes
ENV CUDA_PATH /usr/local/cuda
ENV CUDA_INCLUDE_PATH /usr/local/cuda/include
ENV CUDA_LIBRARY_PATH /usr/local/cuda/lib64

# Ubuntu Packages
RUN apt-get update -y && apt-get install software-properties-common -y && \
    add-apt-repository -y multiverse && apt-get update -y && apt-get upgrade -y && \
    apt-get install -y apt-utils nano vim man build-essential wget sudo && \
    rm -rf /var/lib/apt/lists/*

# Install curl and other dependencies
RUN apt-get update -y && apt-get install -y curl libssl-dev openssl libopenblas-dev \
    libhdf5-dev hdf5-helpers hdf5-tools libhdf5-serial-dev libprotobuf-dev protobuf-compiler git
RUN curl -sk https://raw.githubusercontent.com/torch/distro/master/install-deps | bash && \
    rm -rf /var/lib/apt/lists/*

# Install python3 pip3
RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Python packages we use (or used at one point...)
RUN pip3 install numpy scipy pyyaml matplotlib
RUN pip3 install imageio
RUN pip3 install tensorboard-logger
RUN pip3 install pygame

RUN mkdir /install
WORKDIR /install

# install pysc2
RUN git clone https://github.com/deepmind/pysc2.git /install/pysc2
WORKDIR /install/pysc2
RUN git checkout a9f093493c4c77adb385602790a480e7f238b97d
WORKDIR /install
# Use line below if you want the latest commit
#RUN git clone https://github.com/deepmind/pysc2.git /install/pysc2
RUN pip3 install /install/pysc2/

# install Sacred (from OxWhirl fork)
RUN pip3 install setuptools
RUN git clone https://github.com/oxwhirl/sacred.git /install/sacred && cd /install/sacred && python3 setup.py install

#### -------------------------------------------------------------------
#### install OpenBW and TorchCraft
#### -------------------------------------------------------------------
#
### --------- sdl
#RUN apt-get update && apt-get install -y libsdl2-dev libsdl2-2.0
#
### --------- build OpenBW
#RUN git clone https://github.com/openbw/openbw
#RUN git clone https://github.com/openbw/bwapi
#RUN cd bwapi && mkdir build && cd build \
#    && cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=../../openbw -DOPENBW_ENABLE_UI=$OBW_GUI \
#    && make && make install
#
## -------- TorchCraft
#
#RUN git clone https://github.com/torchcraft/torchcraft.git -b develop --recursive
#
### --------- zmq
#RUN apt-get purge -y libzmq*
#RUN apt-get update && apt-get install -y libtool pkg-config build-essential autoconf automake uuid-dev
#RUN wget https://github.com/zeromq/libzmq/releases/download/v4.2.2/zeromq-4.2.2.tar.gz
#RUN tar xvzf zeromq-4.2.2.tar.gz
## RUN ulimit -n 1000 && apt-get update
#RUN cd zeromq-4.2.2 && ./configure && make install && ldconfig
#
### --------- zstd
## This can only be done on 18.04, so commenting out for now
## RUN apt-get install -y libzstd1-dev zstd
#RUN wget https://github.com/facebook/zstd/archive/v1.1.4.tar.gz
#RUN tar xf v1.1.4.tar.gz
#RUN cd zstd-1.1.4/ && make -j && make install && ldconfig
#
### --------- build BWEnv
#RUN cd torchcraft/BWEnv && mkdir build && cd build \
#    && cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo \
#    && make -j
#
## Installing the python client just in case
### --------- python client deps
#RUN pip3 install pybind11
#
### --------- python client
#RUN cd torchcraft && pip3 install .


## --------- sdl
RUN apt-get update && apt-get install -y libsdl2-dev libsdl2-2.0

## --------- build OpenBW
RUN git clone https://github.com/openbw/openbw
RUN git clone https://github.com/openbw/bwapi
RUN cd bwapi && mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENBW_DIR=../../openbw -DOPENBW_ENABLE_UI=$OBW_GUI \
    && make && make install

# -------- TorchCraft

RUN git clone https://github.com/torchcraft/torchcraft.git -b develop --recursive

## --------- zmq
RUN apt-get purge -y libzmq*
RUN apt-get update && apt-get install -y libtool pkg-config build-essential autoconf automake uuid-dev
RUN wget https://github.com/zeromq/libzmq/releases/download/v4.2.2/zeromq-4.2.2.tar.gz
RUN tar xvzf zeromq-4.2.2.tar.gz
RUN ulimit -n 1000 && apt-get update
RUN cd zeromq-4.2.2 && ./configure && make install && ldconfig

## --------- zstd
# This can only be done on 18.04, so commenting out for now
# RUN apt-get install -y libzstd1-dev zstd
RUN wget https://github.com/facebook/zstd/archive/v1.1.4.tar.gz
RUN tar xf v1.1.4.tar.gz
RUN cd zstd-1.1.4/ && make -j && make install && ldconfig

## --------- build BWEnv
RUN cd torchcraft/BWEnv && mkdir build && cd build && sed -i.bak -e '5d;' ../CMakeLists.txt
RUN cd torchcraft/BWEnv/build && cmake .. -DCMAKE_BUILD_TYPE=relwithdebinfo -DZMQ_LIBRARIES=/usr/local/lib/libzmq.so  \
    && make -j

## --------- python client deps
RUN pip3 install pybind11

## --------- python client
RUN cd torchcraft && pip3 install .

#### -------------------------------------------------------------------
#### install pytorch
#### -------------------------------------------------------------------
RUN pip3 install torch
RUN pip3 install torchvision snakeviz pytest probscale

WORKDIR /pymarl
