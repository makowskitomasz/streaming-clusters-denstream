APP_NAME=streaming-clusters-denstream
DOCKER_IMAGE=$(APP_NAME):latest

.PHONY: help build up run test lint format shell logs

help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker image"
	@echo "  make up         - Run API via docker compose"
	@echo "  make run        - Run API container manually"
	@echo "  make test       - Run pytest via uv"
	@echo "  make lint       - Run ruff"
	@echo "  make format     - Format code with ruff"
	@echo "  make shell      - Open container shell"
	@echo "  make logs       - View docker compose logs"

build:
	docker build -t $(DOCKER_IMAGE) .

up:
	docker compose up --build

run:
	docker run --rm -p 4321:8000 $(DOCKER_IMAGE)

test:
	uv run pytest -q

lint:
	uv run ruff check . --output-format=full --fix

format:
	uv run ruff format .

shell:
	docker run -it --rm $(DOCKER_IMAGE) sh

logs:
	docker compose logs -f
