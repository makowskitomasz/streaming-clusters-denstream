APP_NAME=streaming-clusters-denstream
DOCKER_IMAGE=$(APP_NAME):latest

.PHONY: help build build-backend build-frontend up up-backend up-frontend run run-backend run-frontend test lint lint-fix format shell logs

help:
	@echo "Available commands:"
	@echo "  make build      - Build Docker image"
	@echo "  make build-backend - Build backend Docker image"
	@echo "  make build-frontend - Build frontend Docker image"
	@echo "  make up         - Run API via docker compose"
	@echo "  make up-backend - Run backend via docker compose"
	@echo "  make up-frontend - Run frontend via docker compose"
	@echo "  make run        - Run backend container manually"
	@echo "  make run-backend - Run backend container manually"
	@echo "  make run-frontend - Run Streamlit frontend locally"
	@echo "  make test       - Run pytest via uv"
	@echo "  make lint       - Run ruff and mypy"
	@echo "  make format     - Format code with black and ruff"
	@echo "  make shell      - Open container shell"
	@echo "  make logs       - View docker compose logs"

build:
	@$(MAKE) build-backend
	@$(MAKE) build-frontend

build-backend:
	docker build -t $(DOCKER_IMAGE) -f docker/api/Dockerfile .

build-frontend:
	docker build -t $(APP_NAME)-frontend:latest -f docker/frontend/Dockerfile .

up:
	docker compose up --build

up-backend:
	docker compose up --build api

up-frontend:
	docker compose up --build frontend

run:
	docker run --rm -p 4321:8000 $(DOCKER_IMAGE)

run-backend:
	docker run --rm -p 4321:8000 $(DOCKER_IMAGE)

run-frontend:
	uv run streamlit run src/frontend/app.py

test:
	uv run pytest -q

lint:
	uv run ruff check . --output-format=full
	uv run mypy src/frontend/app.py

lint-fix:
	uv run ruff check . --output-format=full --fix

format:
	uv run black .
	uv run ruff format .

shell:
	docker run -it --rm $(DOCKER_IMAGE) sh

logs:
	docker compose logs -f
