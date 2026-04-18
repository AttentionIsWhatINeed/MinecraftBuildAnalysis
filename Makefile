PYTHON ?= python
PIP ?= $(PYTHON) -m pip
SRC_DIRS := src scripts config

.PHONY: help install build test clean run-data-process run-data-process-visual run-raw-visual run-train run-predict

help:
	@echo "Available targets:"
	@echo "  make install                 Install dependencies"
	@echo "  make build                   Build/compile source files"
	@echo "  make test                    Run unit tests"
	@echo "  make run-data-process        Run data processing pipeline"
	@echo "  make run-data-process-visual Run before/after data visualization"
	@echo "  make run-raw-visual          Run raw data visualization"
	@echo "  make run-train               Run model training"
	@echo "  make run-predict             Run prediction and visualization"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

build:
	$(PYTHON) -m compileall $(SRC_DIRS)

test:
	$(PYTHON) -m pytest -q

run-data-process:
	$(PYTHON) scripts/run_data_process.py

run-data-process-visual:
	$(PYTHON) scripts/run_data_process_visual.py

run-raw-visual:
	$(PYTHON) scripts/run_raw_visual.py

run-train:
	$(PYTHON) scripts/run_train.py

run-predict:
	$(PYTHON) scripts/run_predict.py

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"
