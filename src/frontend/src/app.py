from __future__ import annotations

from time import time

import streamlit as st
from api_client import ApiClient, BackendError
from frontend_sidebar import _build_synthetic_stream_payload, render_sidebar
from frontend_state import (
    HIGH_BATCH_SIZE,
    HIGH_DRIFT_RATE,
    TAB_NAMES,
    SidebarState,
    init_state,
    reset_state,
)
from frontend_stream import call_backend, next_batch_backend
from frontend_tabs import (
    render_tab_current_state,
    render_tab_history_view,
    render_tab_logs,
)

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore[import-untyped]

    AUTOREFRESH_AVAILABLE = True
except ImportError:  # pragma: no cover
    AUTOREFRESH_AVAILABLE = False

    def st_autorefresh(interval: int, key: str) -> None:
        last_key = f"{key}-last"
        now = time()
        last = st.session_state.get(last_key, 0.0)
        if now - last >= interval / 1000:
            st.session_state[last_key] = now
            _rerun()


def _rerun() -> None:
    """Trigger a Streamlit rerun using the available API."""
    rerun = getattr(st, "rerun", None)
    if callable(rerun):
        rerun()
        return
    legacy = getattr(st, "experimental_rerun", None)
    if callable(legacy):  # pragma: no cover
        legacy()


def _get_active_tab_index() -> int:
    """Read the current tab index from query params with bounds checking."""
    qp = st.query_params
    raw = qp.get("tab", "0")
    if isinstance(raw, list):
        raw = raw[0] if raw else "0"
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        idx = 0
    return max(0, min(idx, len(TAB_NAMES) - 1))


def handle_apply_params(state: SidebarState, client: ApiClient) -> None:
    """Apply stream configuration based on sidebar form inputs."""
    if not state.actions.apply_params:
        return

    if state.data_source == "synthetic":
        if state.dynamic_min > state.dynamic_max:
            st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
            return
        payload = _build_synthetic_stream_payload(state)
        try:
            client.configure_stream(payload)
            st.session_state.stream_state = client.get_stream_state()
            st.success("Stream parameters updated.")
        except BackendError as exc:
            st.error(str(exc))
        return

    # nyc_taxi
    try:
        client.configure_nyc_taxi({"file_path": state.nyc_file_path})
        st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
        st.success("Stream parameters updated.")
    except BackendError as exc:
        st.error(str(exc))


def handle_buttons(state: SidebarState, client: ApiClient) -> None:
    """Handle Start/Pause/Reset/Next Batch actions from the sidebar."""
    st.session_state.max_history_points = state.actions.max_history
    st.session_state.ttl_seconds = state.actions.ttl_seconds

    if state.data_source == "synthetic" and (state.batch_size > HIGH_BATCH_SIZE or state.drift_rate > HIGH_DRIFT_RATE):
        st.warning("High values may reduce responsiveness.")

    if state.actions.start:
        if state.data_source == "nyc_taxi":
            try:
                client.configure_nyc_taxi({"file_path": state.nyc_file_path})
                st.session_state.nyc_bounds = client.get_nyc_taxi_bounds()
            except BackendError as exc:
                st.error(str(exc))
                return
            st.session_state.running = True
            st.success("NYC Taxi stream started.")
            return

        # synthetic
        if state.dynamic_min > state.dynamic_max:
            st.error("dynamic_min_clusters must be <= dynamic_max_clusters.")
            return

        payload = _build_synthetic_stream_payload(state)
        try:
            client.configure_stream(payload)
            st.session_state.stream_state = client.get_stream_state()
        except BackendError as exc:
            st.error(str(exc))
            return

        ok, message = call_backend("start", state.params, client)
        if ok:
            st.session_state.running = True
            st.success("Stream started.")
        else:
            st.error(message)

    if state.actions.pause:
        st.session_state.running = False
        if state.data_source == "synthetic":
            call_backend("pause", state.params, client)
        st.info("Stream paused.")

    if state.actions.reset:
        if state.data_source == "nyc_taxi":
            try:
                client.reset_nyc_taxi()
                reset_state()
                st.success("NYC Taxi stream reset.")
            except BackendError as exc:
                st.error(str(exc))
        else:
            ok, message = call_backend("reset", state.params, client)
            if ok:
                reset_state()
                st.success("Stream reset.")
            else:
                st.error(message)

    if state.actions.next_batch:
        try:
            next_batch_backend(state.params, client)
            st.success("Fetched next batch from backend.")
        except BackendError as exc:
            st.error(str(exc))


def handle_autorun(state: SidebarState, client: ApiClient) -> None:
    """Fetch batches automatically while the stream is running."""
    if not st.session_state.running:
        return

    st_autorefresh(interval=int(state.actions.refresh_interval * 1000), key="stream-refresh")
    try:
        next_batch_backend(state.params, client)
        if state.data_source == "synthetic":
            st.session_state.stream_state = client.get_stream_state()
    except BackendError as exc:
        st.error(str(exc))
        st.session_state.running = False


def main() -> None:
    """Render the main Streamlit app."""
    st.set_page_config(page_title="Clustering Dashboard", layout="wide")
    init_state()
    st.session_state.autorefresh_available = AUTOREFRESH_AVAILABLE

    st.title("Clustering Dashboard")
    st.write("Explore DenStream behavior, drift, and batch-level metrics with synthetic data.")

    client = ApiClient()

    sidebar_state = render_sidebar(client)

    st.session_state.max_history_points = sidebar_state.actions.max_history

    # Handle actions
    handle_apply_params(sidebar_state, client)
    handle_buttons(sidebar_state, client)
    handle_autorun(sidebar_state, client)

    # Render tabs
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = TAB_NAMES[_get_active_tab_index()]

    active_name = st.radio(
        "tabs",
        TAB_NAMES,
        key="active_tab",
        horizontal=True,
        label_visibility="collapsed",
    )

    active_idx = TAB_NAMES.index(active_name)
    current_tab = st.query_params.get("tab")
    if isinstance(current_tab, list):
        current_tab = current_tab[0] if current_tab else None
    if current_tab != str(active_idx):
        st.query_params["tab"] = str(active_idx)

    if active_name == "Current State":
        render_tab_current_state()
    elif active_name == "History View":
        render_tab_history_view(
            show_labels=sidebar_state.actions.show_labels,
            show_last_n=sidebar_state.actions.show_last_n,
            last_n=sidebar_state.actions.last_n,
        )
    else:
        render_tab_logs(client)

    st.caption(f"Refresh interval setting: {sidebar_state.actions.refresh_interval} seconds")


if __name__ == "__main__":
    main()
