PY ?= python

.PHONY: install all clean-data features validate eda baseline test clean help

help:
	@echo "make install    - pip install requirements"
	@echo "make all        - full pipeline: clean -> features -> validate -> eda -> baseline"
	@echo "make test       - unit tests (synthetic data) + output checks if data exists"
	@echo "make clean      - remove generated data (keeps reports)"

install:
	$(PY) -m pip install -r requirements.txt pytest

all: clean-data features validate eda baseline

clean-data:
	$(PY) src/01_clean.py

features: 
	$(PY) src/02_build_features.py

validate:
	$(PY) src/04_validate.py

eda:
	$(PY) src/03_eda.py

baseline:
	$(PY) src/05_sanity_baseline.py

test:
	$(PY) -m pytest -q

clean:
	rm -f data/clean/*.parquet data/model_ready/*.parquet data/model_ready/*.csv.gz
