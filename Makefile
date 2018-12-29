.PHONY: build_venv remove_env

VENV_NAME=tac_attn_venv
CONDA_HOME=$(HOME)/anaconda3
CONDA_ENVS=$(CONDA_HOME)/envs

help:
	@echo "build_env - create a fresh conda environment with all requirements already installed"
	@echo "remove_env - delete the previously created virtual environment"

build_venv:
	command -v conda || . $(CONDA_HOME)/etc/profile.d/conda.sh && \
	conda env create -n $(VENV_NAME) -f environment.yml --force && \
	conda activate $(VENV_NAME) && \
	PIP_CONFIG_FILE=$(shell pwd)/pip.conf pip install -r requirements.txt && \
	conda install pytorch torchvision cuda100 -c pytorch && \
	$(CONDA_HOME)/envs/$(VENV_NAME)/bin/python -m spacy download en_core_web_lg

remove_env:
	command -v conda || . $(CONDA_HOME)/etc/profile.d/conda.sh && \
	conda remove --name $(VENV_NAME) --all