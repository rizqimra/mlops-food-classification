PYTHON = python3

.PHONY: all
all: help

setup:
   python -m venv ~/.food-classification
   source ~/.food-classification/bin/activate
   cd .food-classification

install:
   pip install --upgrade pip &&
       pip install -r requirements.txt

download:
	@echo "Downloading dataset..."
	$(PYTHON) download.py

train:
	@echo "Training model..."
	$(PYTHON) train.py

run:
	@echo "Starting Streamlit app..."
	streamlit run streamlit.py

.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make download   - Download dataset"
	@echo "  make train      - Train model"
	@echo "  make run        - Run Streamlit app"
	@echo "  make help       - Show this help message"
