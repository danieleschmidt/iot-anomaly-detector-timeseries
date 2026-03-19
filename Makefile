PYTHON := ~/anaconda3/bin/python3

.PHONY: demo test install

demo:
	$(PYTHON) demo.py

test:
	$(PYTHON) -m pytest tests/ -v

install:
	$(PYTHON) -m pip install -e .
