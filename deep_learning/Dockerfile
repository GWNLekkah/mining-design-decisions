FROM python:3.10-slim-buster as PY
FROM rust:1.68
COPY --from=PY / /
#COPY --from=RUST / /
COPY ./pyproject.toml pyprojec.toml
COPY ./requires.txt requires.txt
COPY ./setup.py setup.py
COPY ./dl_manager dl_manager
RUN apt-get -y update
#RUN apt-get -y install curl
#RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -- -y
#RUN ./rustup-init.sh -y
RUN python3 -m pip install nltk setuptools_rust
#RUN apt-get -y install gcc
#RUN apt-get -y install --reinstall build-essential
RUN python3 -m pip install -r requires.txt
RUN python3 setup.py build_ext --inplace --verbose
CMD ["python3.10", "-m", "dl_manager"]
