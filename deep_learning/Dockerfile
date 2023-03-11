# Step 1: Set up Python and Rust
FROM python:3.10-slim-buster AS PY
FROM rust:1.68-slim-buster
COPY --from=PY / /

# Get a list of all the dependencies required by the dl_manager
RUN python3 -m pip install setuptools_rust nltk
RUN apt-get update
RUN apt-get -y install --reinstall build-essential

# Copy files to their own non-root directory
COPY ./server.crt /app/server.crt
COPY ./server.key /app/server.key
COPY ./setup.py /app/setup.py
COPY ./pyproject.toml /app/pyproject.toml
WORKDIR /app

# Install dependencies needed by dl_manager
RUN python3 setup.py sdist
RUN python3 -m pip install -r dl_manager.egg-info/requires.txt

# Build extension modules
COPY ./dl_manager /app/dl_manager
RUN python3 setup.py build_ext --inplace --verbose

# Setup start-up command
CMD ["python3.10", "-m", "dl_manager", "serve", "--keyfile", "server.key", "--certfile", "server.crt"]
