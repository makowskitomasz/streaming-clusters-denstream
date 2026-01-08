# Evolving Cluster Detection in Data Streams

## Platform Overview
This project delivers a full-stack platform for detecting evolving clusters in streaming data. The backend exposes a FastAPI service that ingests, clusters, and analyzes real-time streams, while a Streamlit dashboard visualizes the evolution of clusters, metrics, and configuration states. The architecture is oriented to experimentation and teaching, balancing online DenStream processing with offline HDBSCAN baselines.

---

## Project Goal
The system enables analysis of continuously arriving data and provides a controlled environment to compare DenStream (online) with HDBSCAN (offline snapshot baseline). Researchers can study concept drift, noise, stability, and clustering quality over time by replaying synthetic or real-world streams and observing how both algorithms respond to shifting distributions.

---

## Technology Stack
- **Python**: Version 3.12 across backend and frontend logic.
- **FastAPI**: REST backend that exposes ingestion, configuration, and monitoring endpoints with async processing.
- **Streamlit**: Python UI for real-time dashboards, controls, and visual analytics.
- **uv**: Dependency management and virtual environments via `pyproject.toml` and `uv.lock`.
- **Docker**: Containerized backend and frontend execution.
- **UMAP / scikit-learn / HDBSCAN**: Dimensionality reduction, feature preprocessing, and offline clustering baseline.
- **River (DenStream)**: Online micro-cluster maintenance for streaming data.
- **matplotlib / Plotly**: Visual layers for cluster trajectories and metrics.
- **pytest**: Unit test suite covering services, controllers, and data models.
- **mypy**: Static type checker verifying type annotations across backend modules.
- **ruff**: Static analysis and linting aligned with PEP8.
- **GitHub Actions CI**: Automation for linting and tests on every push.

---

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

---

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
- `src/frontend/src/`: Streamlit UI modules, API client, and visualization logic.


---

## Backend Runbook (Containerized)

1. **Build and start the backend service**

   ```bash
   make up-backend
   ```

2. **Wait for the service to become available**
   You should see a log line similar to:

   ```
   Uvicorn running on http://0.0.0.0:8000
   ```

3. **Access the API**

   * Swagger UI: `http://localhost:8000/docs`
   * API base URL: `http://localhost:8000`

4. **View backend logs**

   ```bash
   make logs
   ```

5. **Stop the backend**

   ```bash
   docker compose down
   ```

---

## Frontend Runbook (Containerized)

1. Build and start the frontend service

  ```bash
  make up-frontend
  ```
2. Access the dashboard
 
    Open http://localhost:8501 in your browser.
  
3. Stop the frontend

  ```bash
  docker compose down
  ```

---

## CI/CD
GitHub Actions runs linting and tests on every push using `uv` for dependency
resolution. The pipeline executes `ruff`, `mypy`, and `pytest` to ensure
code quality and correctness.

---

## Linting, Formatting & Tests

Common development commands:
- `make test` – run pytest
- `make lint` – run ruff and mypy
- `make format` – apply black and ruff formatting
- `make check` – run lint + tests
- `make coverage` – run tests with coverage enforcement

All commands rely on `uv` and `pyproject.toml` as the single source of truth.

---

## Git Hooks
Use `make hooks` once to install local hooks:
- pre-commit runs lint/format only
- pre-push runs lint + tests
- emergency bypass: `git push --no-verify`

---

## Sample Data Flow
`Stream → Controller → DenStream → Cluster Model → Metrics → API / Streamlit`

---

## System Requirements
- Python 3.12
- uv
- Docker & Docker Compose (optional, for containerized execution)
- ≥ 8 GB RAM recommended for large streaming experiments (e.g. NYC Taxi)

