# Step 1: Set up Python and Rust
FROM python:3.10-slim-buster AS PY
FROM rust:1.68-slim-buster
COPY --from=PY / /

# Copy files to their own non-root directory
COPY ./setup.py /app/setup.py
COPY ./pyproject.toml /app/pyproject.toml
COPY ./dl_manager /app/dl_manager
WORKDIR /app

# Get a list of all the dependencies required by the dl_manager
RUN python3 -m pip install setuptools_rust nltk
RUN python3 setup.py sdist
RUN apt-get update
RUN apt-get -y install --reinstall build-essential

# Install dependencies needed by dl_manager
RUN python3 -m pip install -r dl_manager.egg-info/requires.txt

# Build extension modules
RUN python3 setup.py build_ext --inplace --verbose

# Setup start-up command
CMD ["python3.10", "-m", "dl_manager", "serve"]
