APP_NAME=streaming-clusters-denstream
DOCKER_IMAGE_BACKEND=$(APP_NAME)-backend:latest
DOCKER_IMAGE_FRONTEND=$(APP_NAME)-frontend:latest

.PHONY: help build-backend build-frontend up-backend up-frontend run-backend run-frontend test lint lint-fix format shell logs

help:
	@echo "Available commands:"
	@echo "  make build-backend - Build backend Docker image"
	@echo "  make build-frontend - Build frontend Docker image"
	@echo "  make up-backend - Run backend via docker compose"
	@echo "  make up-frontend - Run frontend via docker compose"
	@echo "  make run-backend - Run backend container manually"
	@echo "  make run-frontend - Run frontend container manually"
	@echo "  make test       - Run pytest via uv"
	@echo "  make lint       - Run ruff and mypy"
	@echo "  make format     - Format code with black and ruff"
	@echo "  make shell      - Open container shell"
	@echo "  make logs       - View docker compose logs"

build-backend:
	docker build -t $(DOCKER_IMAGE_BACKEND) -f docker/api/Dockerfile .

build-frontend:
	docker build -t $(DOCKER_IMAGE_FRONTEND) -f docker/frontend/Dockerfile .

up-backend:
	docker compose up --build api

up-frontend:
	docker compose up --build frontend

run-backend:
	docker run --rm -p 4321:8000 $(DOCKER_IMAGE_BACKEND)

run-frontend:
	docker run --rm -p 8501:8501 $(DOCKER_IMAGE_FRONTEND)

test:
	uv run pytest -q

lint:
	uv run ruff check . --output-format=full
	uv run mypy src/frontend

lint-fix:
	uv run ruff check . --output-format=full --fix

format:
	uv run black .
	uv run ruff format .

shell:
	docker run -it --rm $(DOCKER_IMAGE_BACKEND) sh

logs:
	docker compose logs -f
