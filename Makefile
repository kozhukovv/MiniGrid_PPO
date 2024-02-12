SHELL := /bin/bash

MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

DOCKER_COMPOSE_FILES = docker/docker-compose.yaml

PARAMETERS := ROOT_DIR=$(ROOT_DIR) \
			  DOCKER_COMPOSE_FILES=$(DOCKER_COMPOSE_FILES)


prepare-git-repo:
	@echo "Install pre-commit"
	pip3 install pre-commit
	pre-commit install

build-docker:
	# @echo "Build docker"
	# docker compose -f $(DOCKER_COMPOSE_FILES) build

 run-docker:
	 @echo "Run docker"
	 docker compose -f $(DOCKER_COMPOSE_FILES) up

test:
	pytest --cov=. tests/ --cov-append

cov-badge:
	coverage-badge
