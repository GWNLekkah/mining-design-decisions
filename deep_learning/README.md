# Deep Learning Manager -- How To Use 

---

This document gives an overview how to use the deep learning (`dl_manager`) tool.

---

## Installation

There are a number of ways to install and run the `dl_manager`.
The `dl_mananger` can either be run directly, or through Docker.

### Running Directly 
In order to run the `dl_manager` directly, a Rust compiler must be installed 
in order to compile the acceleration modules.

Installation instructions for Rust can be found here:
https://www.rust-lang.org/tools/install

Next, optionally, CUDA can be installed in order to make training faster.
Note that this can only be done on machines with NVIDIA GPUs installed. 
Instructions can be found at https://www.tensorflow.org/install/pip .
The current version of the pipeline requires CUDA 11.2.0 and CuDNN 8.6.0.

Next, modules necessary for installation must be installed. This can be done 
using 

```shell 
python -m pip install setuptools_rust nltk 
```

Next, install the dependencies of the `dl_manager` by running 

```shell 
python setup.py sdist 
python -m pip install -r dl_manager.egg-info/requires.txt
```

Finally, the acceleration module can now be build, using 

```shell 
python setup.py build_ext --inplace 
```

When using the pipeline in webserver mode, (self-signed) SSL certificates 
will be required. This can be done using 

```shell 
openssl req -new -x509 -nodes -sha256 -out server.crt -keyout server.key
```

Finally, the pipeline can be started in webserver mode using 

```shell 
python3 -m dl_manager serve --keyfile server.key --certfile server.crt
```

### Running through Docker

---

## Running - Main Help Message 

The basic command to run the manager is using `python __main__.py`. When specifying the 
`-h` flag, the following output is printed:

```
sage: __main__.py [-h] {list,hyperparams,generator-params,make-features,run,combination-strategies,run_analysis} ...

options:
  -h, --help            show this help message and exit

Sub-commands:
  {list,hyperparams,generator-params,make-features,run,combination-strategies,run_analysis}
    list                List options for various CLI options
    hyperparams         View hyper-parameters for a classifier
    generator-params    View parameters for a feature generator
    make-features       Generate a collection of features
    run                 Train a classifier and store the results
    combination-strategies
                        Give a comprehensive overview of all available model combination strategies.
    run_analysis        Analyze the results of deep learning runs
```

This output give a list of multiple possible commands one can run. 
Links to more explanation can be found below:

- [list](docs/list.md) - A utility to elaborate on options for some parameters.
- [hyperparams](docs/hyperparams.md) - Get hyperparameters for a classifier 
- [generator-params](docs/generator-param.md) - Parameters for feature generators 
- [make-features](docs/make-features.md) - Generate Feature Vectors 
- [run](docs/run.md) - Train and test a classifier
- [combination-strategies](docs/combination-strategies.md)
- [run_analysis](docs/analysis.md) - Analyze the results of the `run` command