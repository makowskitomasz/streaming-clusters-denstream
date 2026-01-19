from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from .api_client import ApiClient, BackendError, StreamParams
from .frontend_state import (
    NYC_EPSILON_DEFAULT_M,
    NYC_EPSILON_MAX_M,
    NYC_EPSILON_MIN_M,
    NYC_EPSILON_STEP_M,
    NYC_MU_DEFAULT,
    NYC_MU_MAX,
    NYC_MU_MIN,
    NYC_MU_STEP,
    SYN_EPSILON_MAX,
    SYN_EPSILON_MIN,
    SYN_EPSILON_STEP,
    SYN_MU_DEFAULT,
    SYN_MU_MAX,
    SYN_MU_MIN,
    SYN_MU_STEP,
    SidebarState,
    UiActions,
)

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


def _denstream_defaults(data_source: str) -> dict[str, float]:
    """Return UI defaults for DenStream based on data source."""
    if data_source == "nyc_taxi":
        return {
            "epsilon": NYC_EPSILON_DEFAULT_M,  # meters
            "beta": 0.30,
            "mu": NYC_MU_DEFAULT,
        }
    return {
        "epsilon": 0.50,  # synthetic coordinate units
        "beta": 0.50,
        "mu": SYN_MU_DEFAULT,
    }


def _epsilon_input(den_form: DeltaGenerator, data_source: str, default_epsilon: float) -> float:
    """Render the epsilon input with NYC or synthetic ranges."""
    if data_source == "nyc_taxi":
        return float(
            den_form.number_input(
                "epsilon_meters",
                min_value=NYC_EPSILON_MIN_M,
                max_value=NYC_EPSILON_MAX_M,
                value=default_epsilon,
                step=NYC_EPSILON_STEP_M,
                format="%.0f",
                help="Neighborhood radius in meters (requires clustering in projected meters).",
            ),
        )
    return float(
        den_form.number_input(
            "epsilon",
            min_value=SYN_EPSILON_MIN,
            max_value=SYN_EPSILON_MAX,
            value=default_epsilon,
            step=SYN_EPSILON_STEP,
            format="%.3f",
        ),
    )


def _mu_input(den_form: DeltaGenerator, data_source: str, default_mu: float) -> float:
    """Render the mu input with NYC or synthetic ranges."""
    if data_source == "nyc_taxi":
        return float(
            den_form.number_input(
                "mu",
                min_value=NYC_MU_MIN,
                max_value=NYC_MU_MAX,
                value=default_mu,
                step=NYC_MU_STEP,
                format="%.0f",
            ),
        )
    return float(
        den_form.number_input(
            "mu",
            min_value=SYN_MU_MIN,
            max_value=SYN_MU_MAX,
            value=default_mu,
            step=SYN_MU_STEP,
            format="%.1f",
        ),
    )


def _build_synthetic_stream_payload(state: SidebarState) -> dict[str, object]:
    """Build the synthetic stream configuration payload."""
    payload: dict[str, object] = {
        "points_per_cluster": state.batch_size,
        "drift": state.drift_rate,
    }
    if state.dynamic_enabled:
        payload.update(
            {
                "dynamic_enabled": True,
                "dynamic_interval": state.dynamic_interval,
                "dynamic_min_clusters": state.dynamic_min,
                "dynamic_max_clusters": state.dynamic_max,
            },
        )
    return payload


def render_sidebar(client: ApiClient) -> SidebarState:
    """Render sidebar controls and return collected state."""
    """
    Renders the entire sidebar and returns:
    - StreamParams used by the backend calls
    - UiActions flags + display options
    - Key widget values needed for action handlers
    """
    with st.sidebar:
        st.subheader("Stream controls")

        data_source = st.selectbox(
            "data_source",
            ["synthetic", "nyc_taxi"],
            index=0 if st.session_state.data_source == "synthetic" else 1,
        )
        st.session_state.data_source = data_source

        params_form = st.form("stream-params")

        batch_size = 150
        drift_rate = 0.2
        nyc_file_path = st.session_state.nyc_file_path

        refresh_interval = params_form.slider(
            "update_interval_seconds",
            0.1,
            5.0,
            0.5,
            step=0.1,
        )

        if data_source == "nyc_taxi":
            nyc_file_path = params_form.text_input(
                "nyc_file_path",
                value=st.session_state.nyc_file_path,
            )
            st.session_state.nyc_file_path = nyc_file_path
        else:
            batch_size = params_form.slider("points_per_cluster", 10, 1000, 150, step=10)
            drift_rate = params_form.slider("drift_rate", 0.0, 2.0, 0.2, step=0.05)

        apply_params = params_form.form_submit_button("Apply")

        ttl_seconds = st.slider("point_ttl_seconds (0 = off)", 0.0, 30.0, 5.0, step=0.5)
        st.session_state.ttl_seconds = ttl_seconds

        dynamic_enabled = False
        dynamic_interval = 15
        dynamic_min = 2
        dynamic_max = 6

        if data_source == "synthetic":
            st.session_state.point_mode = True
            points_per_tick = st.slider("points_per_tick", 1, 200, 25, step=1)
            st.session_state.points_per_tick = points_per_tick

            dynamic_enabled = st.checkbox("Dynamic clusters", value=False)
            dynamic_interval = st.slider("dynamic_interval", 5, 50, 15, step=5)
            dynamic_min = st.slider("dynamic_min_clusters", 1, 10, 2, step=1)
            dynamic_max = st.slider("dynamic_max_clusters", 2, 12, 6, step=1)

        max_history = st.slider("max_history_points", 50, 500, 200, step=10)
        show_last_n = st.checkbox("Show only last N steps", value=True)
        last_n = st.slider("history_window", 20, 200, 50, step=10)
        show_labels = st.checkbox("Show centroid labels", value=True)

        start = st.button("Start Stream", width="stretch")
        pause = st.button("Pause", width="stretch")
        reset = st.button("Reset", width="stretch")
        next_batch = st.button("Next Batch", width="stretch")

        status = "Running" if st.session_state.running else "Paused"
        st.caption(f"Status: {status}")
        st.caption(f"Backend: {st.session_state.backend_status}")

        if st.session_state.stream_state:
            current_clusters = st.session_state.stream_state.get("n_clusters", "—")
            dyn_enabled = st.session_state.stream_state.get("dynamic_enabled", "—")
            st.caption(f"Stream clusters: {current_clusters} | dynamic: {dyn_enabled}")

        if st.session_state.running and not st.session_state.get("autorefresh_available", True):
            st.warning("Auto-refresh unavailable. Install streamlit-autorefresh to animate.")

        with st.expander("DenStream parameters", expanded=False):
            den_form = st.form("denstream-params")

            defaults = _denstream_defaults(data_source)

            decay_factor = float(
                den_form.number_input(
                    "decay_factor",
                    min_value=0.001,
                    max_value=0.2,
                    value=0.01,
                    step=0.001,
                    format="%.4f",
                ),
            )

            epsilon = _epsilon_input(den_form, data_source, defaults["epsilon"])

            beta = float(
                den_form.number_input(
                    "beta",
                    min_value=0.1,
                    max_value=0.9,
                    value=defaults["beta"],
                    step=0.05,
                    format="%.2f",
                ),
            )

            mu = _mu_input(den_form, data_source, defaults["mu"])

            n_samples_init = int(
                den_form.number_input(
                    "n_samples_init",
                    min_value=10,
                    max_value=1000,
                    value=200,
                    step=10,
                ),
            )

            stream_speed = int(
                den_form.number_input(
                    "stream_speed",
                    min_value=1,
                    max_value=500,
                    value=50,
                    step=5,
                ),
            )
            apply_denstream = den_form.form_submit_button("Apply DenStream settings")

            if apply_denstream:
                try:
                    client.configure_denstream(
                        {
                            "decay_factor": decay_factor,
                            "epsilon": epsilon,
                            "beta": beta,
                            "mu": mu,
                            "n_samples_init": n_samples_init,
                            "stream_speed": stream_speed,
                        },
                    )
                    st.success("DenStream configuration updated.")
                except BackendError as exc:
                    st.error(str(exc))

    refresh_interval_seconds = int(max(1, round(refresh_interval)))
    if data_source == "nyc_taxi":
        params = StreamParams(
            batch_size=batch_size if data_source == "synthetic" else 0,
            drift_rate=drift_rate if data_source == "synthetic" else 0.0,
            update_interval_seconds=refresh_interval_seconds,
        )
    else:
        params = StreamParams(
            batch_size=batch_size,
            drift_rate=drift_rate,
            update_interval_seconds=refresh_interval_seconds,
        )

    actions = UiActions(
        apply_params=apply_params,
        start=start,
        pause=pause,
        reset=reset,
        next_batch=next_batch,
        refresh_interval=refresh_interval,
        max_history=max_history,
        ttl_seconds=ttl_seconds,
        show_last_n=show_last_n,
        last_n=last_n,
        show_labels=show_labels,
        log_limit=int(st.session_state.get("log_limit", 200)),
        auto_refresh_logs=bool(st.session_state.get("auto_refresh_logs", True)),
        refresh_logs_clicked=False,
    )
    return SidebarState(
        params=params,
        actions=actions,
        data_source=data_source,
        nyc_file_path=nyc_file_path,
        batch_size=batch_size,
        drift_rate=drift_rate,
        dynamic_enabled=dynamic_enabled,
        dynamic_interval=dynamic_interval,
        dynamic_min=dynamic_min,
        dynamic_max=dynamic_max,
    )
