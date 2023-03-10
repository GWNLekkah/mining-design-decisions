FROM rust:1.68
#RUN apt-get install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get upgrade -y
RUN apt-get -y install python3.10
COPY ./pyproject.toml pyprojec.toml
COPY ./requires.txt requires.txt
COPY ./setup.py setup.py
COPY ./dl_manager dl_manager
RUN apt-get -y install python3-pip
RUN python3 -m pip install nltk setuptools_rust
RUN python3 -m pip install -r requires.txt
RUN python3 setup.py build_ext --inplace --verbose
CMD ["python3.10", "-m", "dl_manager"]
