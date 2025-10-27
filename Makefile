ENV_NAME=streaming_clusters
ENV_FILE=environment.yml

.PHONY: help env update activate run test clean lint notebooks

help:
	@echo "Available commands:"
	@echo "  make env        - Create Conda environment"
	@echo "  make update     - Update Conda environment"
	@echo "  make activate   - Show activation command"
	@echo "  make run        - Run main.py"
	@echo "  make test       - Run unit tests"
	@echo "  make lint       - Run ruff"

env:
	conda env create -f $(ENV_FILE)

update:
	conda env update --file $(ENV_FILE) --prune

activate:
	@echo "To activate the environment, run:"
	@echo "  conda activate $(ENV_NAME)"

run:
	python src/clustering_api/src/main.py

test:
	pytest tests/ --maxfail=1 --disable-warnings -q

lint:
	ruff check . --output-format=full --fix
