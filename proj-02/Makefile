# Minimal Makefile for a conda env defined in environment.yml
SHELL := /bin/bash
ENV_FILE ?= environment.yml
ENV_NAME ?= project-env
CONDA_BASE := $(shell conda info --base 2>/dev/null)

.PHONY: env-create env-update env-remove

env-create:
	conda env create -f $(ENV_FILE)

env-update:
	conda env update -f $(ENV_FILE) --prune

env-remove:
	conda env remove -n $(ENV_NAME)
