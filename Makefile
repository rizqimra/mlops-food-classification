PYTHON = python3

.PHONY: all
all: install test

setup:
	$(PYTHON) -m venv ~/.food-classification
	. ~/.food-classification/bin/activate
	cd ~/.food-classification

install:
	pip install --upgrade pip && pip install -r requirements.txt

download:
	@echo "Downloading dataset..."
	$(PYTHON) ./src/download.py

train:
	@echo "Training model..."
	$(PYTHON) ./src/train.py

test:
	@echo "Evaluating model..."
	$(PYTHON) ./src/test.py

run:
	@echo "Starting Streamlit app..."
	streamlit run src/streamlit.py

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make setup      - Set up virtual environment"
	@echo "  make install    - Install dependencies"
	@echo "  make download   - Download dataset"
	@echo "  make train      - Train model"
	@echo "  make test       - Test model"
	@echo "  make run        - Run Streamlit app"
	@echo "  make help       - Show this help message"
