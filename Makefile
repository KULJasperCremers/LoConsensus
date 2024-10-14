ifeq ($(OS), Windows_NT)
	MAKE_OS:=Windows
	RMDIR_CMD=rmdir /Q /S
else
	MAKE_OS:=Linux
	RMDIR_CMD=rm -rf
endif

PYTHON_VERSION=3.12
VENV_NAME=.venv
REPO_NAME=$(notdir $(abspath ./))

BUILD_DIR=./_build
BUILD_WHEEL_DIR=$(BUILD_DIR)/wheel
BUILD_TEST_DIR=$(BUILD_DIR)/test

NAMESPACE_PACKAGES = $(dir $(wildcard ./src/*/pyproject.toml))
PACKAGE_INIT_FILES = $(dir $(wildcard ./src/*/__init__.py))
BUILD_FOLDER_FILES = $(wildcard ./_build/wheel/*)

ifeq ($(MAKE_OS), Windows)
	CREATE_ENV_CMD=py -$(PYTHON_VERSION) -m venv $(VENV_NAME)
	PYTHON=$(VENV_NAME)\Scripts\python
	ACTIVATE=$(VENV_NAME)\Scripts\activate
	TOML_ADAPT=$(VENV_NAME)\Scripts\toml-adapt
	CMDSEP=&
else
	CREATE_ENV_CMD=python$(PYTHON_VERSION) -m venv $(VENV_NAME)
	PYTHON=$(VENV_NAME)/bin/python
	ACTIVATE=source $(VENV_NAME)/bin/activate
	TOML_ADAPT=$(VENV_NAME)/bin/toml-adapt
	CMDSEP=;
endif

RUN_MODULE = $(PYTHON) -m
PIP = $(RUN_MODULE) pip

install: create-env install-requirements 

create-env:
	$(info MAKE: Initializing environment in .venv ...)
	$(CREATE_ENV_CMD)
	$(PIP) install --upgrade "pip>=24.2" wheel
	$(PIP) install artifacts-keyring==0.3.*

install-requirements:
	$(info MAKE: Installing development requirements ...)
	$(PIP) install -r requirements.txt