# Start from a nice and fresh Ubuntu installation
#FROM ubuntu:20.04
FROM nvidia/cuda:11.2.0-cudnn8-runtime AS CUDA

# Remove NVIDIA repositories because they mess with apt-get update
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN rm /etc/apt/sources.list.d/cuda.list

# Install Python3.10
RUN apt-get update
RUN apt-get -y install software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get -y install python3.10-dev

# Install Cargo
RUN apt-get -y install curl
RUN curl https://sh.rustup.rs -sSf > rustup-init.sh
RUN sh rustup-init.sh -y
ENV PATH="$PATH:/root/.cargo/bin"

# Check that all software is installed correctly
RUN python3.10 --version
RUN cargo --version
RUN find | grep cudart

#
COPY ./server.crt /app/server.crt
COPY ./server.key /app/server.key
COPY ./setup.py /app/setup.py
COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
#RUN apt-get -y install python3-pip
#RUN apt-get -y install python3.10-distutils
RUN python3.10 -m pip install setuptools_rust nltk
RUN apt-get -y install --reinstall build-essential
RUN python3.10 setup.py sdist
RUN python3.10 -m pip install -r dl_manager.egg-info/requires.txt

COPY ./dl_manager /app/dl_manager
RUN python3.10 setup.py build_ext --inplace --verbose

# Setup start-up command
CMD ["python3.10", "-m", "dl_manager", "serve", "--keyfile", "server.key", "--certfile", "server.crt"]
