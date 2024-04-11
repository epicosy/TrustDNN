# A framework for evaluating tools that reason about the trustworthiness of the DNN's predictions.

## Installation
TrustDNN is implemented in Python 3.10. To install the required packages, run:

```shell
#Optional: Create a virtual environment
$ python3.10 -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ export TRUSTDNN_DIR=~/.trustdnn
$ mkdir $TRUSTDNN_DIR
$ cp -r config $TRUSTDNN_DIR
```


## Development

This project includes a number of helpers in the `Makefile` to streamline common development tasks.

### Environment Setup

The following demonstrates setting up and working with a development environment:

```
### create a virtualenv for development

$ make virtualenv

$ source env/bin/activate


### run trustdnn cli application

$ trustdnn --help


### run pytest / coverage

$ make test
```


### Releasing to PyPi

Before releasing to PyPi, you must configure your login credentials:

**~/.pypirc**:

```
[pypi]
username = YOUR_USERNAME
password = YOUR_PASSWORD
```

Then use the included helper function via the `Makefile`:

```
$ make dist

$ make dist-upload
```

## Deployments

### Docker

Included is a basic `Dockerfile` for building and distributing `TrustDNN`,
and can be built with the included `make` helper:

```
$ make docker

$ docker run -it trustdnn --help
```
