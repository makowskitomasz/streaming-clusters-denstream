# Evolving Cluster Detection in Data Streams

## Platform Overview
This project delivers a full-stack platform for detecting evolving clusters in streaming data. The backend exposes a FastAPI service that ingests, clusters, and analyzes real-time streams, while a Streamlit dashboard visualizes the evolution of clusters, metrics, and configuration states. The architecture is oriented to experimentation and teaching, balancing online DenStream processing with offline HDBSCAN baselines.

## Project Goal
The system enables analysis of continuously arriving data and provides a controlled environment to compare DenStream (online) with HDBSCAN (offline snapshot baseline). Researchers can study concept drift, noise, stability, and clustering quality over time by replaying synthetic or real-world streams and observing how both algorithms respond to shifting distributions.

## Technology Stack
- **Python**: Version 3.12 across backend and frontend logic.
- **FastAPI**: REST backend that exposes ingestion, configuration, and monitoring endpoints with async processing.
- **Streamlit**: Python UI for real-time dashboards, controls, and visual analytics.
- **UMAP / scikit-learn / HDBSCAN**: Dimensionality reduction, feature preprocessing, and offline clustering baseline.
- **River (DenStream)**: Online micro-cluster maintenance for streaming data.
- **matplotlib / Plotly**: Visual layers for cluster trajectories and metrics.
- **pytest**: Unit test suite covering services, controllers, and data models.
- **mypy**: Static type checker verifying type annotations across backend modules.
- **ruff**: Static analysis and linting aligned with PEP8.
- **GitHub Actions CI**: Automation for linting and tests on every push.
- **environment.yml**: Conda specification for reproducible environments.

## Core Functionality
- **Synthetic stream generation** with tunable cluster counts, drift intensity, noise ratio, and batch size for reproducible experiments.
- **NYC Taxi ingestion** via an adapter that batches historical data into stream-like chunks for real-world benchmarks.
- **DenStream module** that updates micro-clusters online, maintains potential/outlier clusters, and materializes macro clusters on demand.
- **HDBSCAN module** that snapshots buffered data, applies offline clustering, and serves as a stability baseline.
- **Unified models** (`DataPoint`, `ClusterPoint`, `Cluster`) to standardize payloads between API, services, and UI layers.
- **Metrics engine** calculating silhouette, purity, drift score, noise ratio, and active-cluster counts per batch.
- **Streamlit dashboard** that renders embeddings, cluster timelines, and metric traces while exposing configuration controls.
- **REST API** covering stream lifecycle, clustering endpoints, configuration changes, and diagnostic metrics.
- **Reset and reconfiguration** capabilities that clear stateful services and apply new stream or clustering settings without restarts.

## Directory Layout
```
src/
  clustering_api/
    src/
      controllers/
      services/
      adapters/
      models/
      utils/
    tests/
  frontend/
    src/
    tests/
```
- `src/clustering_api/src/controllers/`: FastAPI routers orchestrating endpoints, validation, and lifecycle hooks.
- `src/clustering_api/src/services/`: Business logic for stream generation, DenStream, HDBSCAN, metrics, and orchestration.
- `src/clustering_api/src/adapters/`: Data-source adapters (e.g., NYC Taxi loaders) and stream abstractions.
- `src/clustering_api/src/models/`: Pydantic schemas shared between API and frontend.
- `src/clustering_api/src/utils/`: Helpers for configuration, logging, sampling, and serialization.
- `src/clustering_api/tests/`: pytest suite for controllers, services, and models.
- `frontend/streamlit_app/`: Streamlit UI modules, component definitions, and visualization logic.

## Backend Runbook
### Local (Conda/uv)
1. **Install dependencies**: `conda env create -f environment.yml`.
2. **Activate environment**: `conda activate streaming-clusters` (replace with the name defined in `environment.yml`).
3. **Install Python deps**: `uv sync` (uses `pyproject.toml` + `uv.lock`).
4. **Start FastAPI**: `uv run uvicorn clustering_api.src.main:app --host 0.0.0.0 --port 8000` or run `python src/clustering_api/src/main.py`.
5. **Inspect Swagger UI**: open `http://localhost:8000/docs` to explore REST endpoints and schemas.

### Containerized
1. **Build & run**: `docker compose up --build` (uses the provided Dockerfile pinned to `pyproject.toml`/`uv.lock`).
2. Wait for `Uvicorn running on http://0.0.0.0:8000` in the logs.
3. Open `http://localhost:8000/docs` for Swagger or hit the API directly.
4. Stop everything with `Ctrl+C` or `docker compose down`.

## Frontend Runbook (Streamlit)
1. Launch the UI: `streamlit run src/frontend/src/app.py`.
2. Use the sidebar to pick data sources, configure drift/noise parameters, start or reset streams, and track cluster evolution and metrics in real time.

## CI/CD
GitHub Actions runs tests and linting on every push. The workflow provisions a Conda environment, installs dependencies, executes `ruff`, and runs `pytest`. No production deployment is performed; the pipeline ensures local teaching experiments remain consistent.

## Unit Testing
pytest validates the core services, API layers, and shared models that drive the streaming workflow. Run `make test` or `pytest` from the repository root after activating the environment.

## Linting & Formatting
`ruff` enforces PEP8-compatible style as part of local development and CI, while `Black` provides deterministic formatting. Static typing is checked with `mypy`. Run `make lint`, `make format`, and `make typecheck` (or invoke `ruff`, `black`, and `mypy` directly) before pushing changes.

## Git Hooks
Use `make hooks` once to install local hooks:
- pre-commit runs lint/format only
- pre-push runs lint + tests
- emergency bypass: `git push --no-verify`

## Sample Data Flow
`Stream → Controller → DenStream → Cluster Model → Metrics → API / Streamlit`

## System Requirements
Python 3.12, Conda, and sufficient memory to hold buffered batches (>=8 GB RAM suggested for NYC Taxi experiments).
