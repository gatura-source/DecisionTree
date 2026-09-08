# Python interpreter
PYTHON ?= python3

.PHONY: all help install run test lint format clean

help:
	@echo "Available targets:"
	@echo "  install   Install runtime dependencies (requirements.txt)"
	@echo "  run       Run the demo script (ds.py)"
	@echo "  test      Run the test suite"
	@echo "  lint      Lint the source code"
	@echo "  format    Auto-format the source code"
	@echo "  clean     Remove build/cache artifacts"

all: install

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) ds.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m flake8 ds.py tests

format:
	$(PYTHON) -m black ds.py tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage *.egg-info
