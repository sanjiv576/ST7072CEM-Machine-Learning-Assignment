from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Himalayan Glacier 2030 Simulator",
    layout="wide",
    page_icon="🏔️",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def project_root() -> Path:
    """Return the repository root regardless of Streamlit's working directory."""
    app_dir = Path(__file__).resolve().parent
    candidates = (app_dir.parent, Path.cwd(), app_dir)
    for candidate in candidates:
        if (candidate / "pkl").is_dir() and (candidate / "final_datasets").is_dir():
            return candidate
    return app_dir.parent


ROOT = project_root()
DATA_DIR = ROOT / "final_datasets" / "clean_datasets"
MODEL_DIR = ROOT / "pkl"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load the prepared dataset, preferring Parquet and falling back to CSV."""
    parquet_path = DATA_DIR / "1_data_preparation_dataset.parquet"
    csv_path = DATA_DIR / "1_data_preparation_dataset.csv"

    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except (ImportError, ValueError, OSError):
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        "No prepared dataset was found. Expected either "
        f"'{parquet_path}' or '{csv_path}'."
    )


@st.cache_resource
def load_artifacts():
    """Load the preprocessing objects and trained classifier."""
    paths = {
        "collinear features": MODEL_DIR / "collinear_dropped.pkl",
        "scaler": MODEL_DIR / "scaler.pkl",
        "cluster model": MODEL_DIR / "kmeans_model.pkl",
        "PCA model": MODEL_DIR / "pca_model.pkl",
        "classifier": MODEL_DIR / "final_xgb_model.pkl",
    }
    missing = [f"{name} ({path})" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following deployment artifacts are missing: " + ", ".join(missing)
        )

    return (
        load_dataset(),
        joblib.load(paths["collinear features"]),
        joblib.load(paths["scaler"]),
        joblib.load(paths["cluster model"]),
        joblib.load(paths["PCA model"]),
        joblib.load(paths["classifier"]),
    )


try:
    df, collinear_dropped, scaler, kmeans, pca, xgb_model = load_artifacts()
except (
    AttributeError,
    FileNotFoundError,
    ImportError,
    ModuleNotFoundError,
    OSError,
    TypeError,
    ValueError,
) as error:
    st.error(f"Unable to load the application data or model artifacts: {error}")
    st.info(
        "Deploy the `final_datasets/clean_datasets/` data files and `pkl/` "
        "artifacts alongside `notebooks/app.py`."
    )
    st.stop()


st.title("🏔️ Himalayan Glacier 2030 Simulator")
st.markdown("**A Machine Learning Early-Warning System for Climate Hazards in Nepal**")
st.markdown("---")

st.sidebar.header("⚙️ Climate Simulation Controls")
st.sidebar.markdown("Adjust the meteorological drivers to simulate the year 2030.")
temp_increase = st.sidebar.slider(
    "Temperature Increase (°C)",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Increase the 2m surface temperature across all glaciers.",
)
snow_decrease = st.sidebar.slider(
    "Snowfall Reduction (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="Simulate reduced monsoonal snowfall.",
)
run_sim = st.sidebar.button("Run 2030 Simulation")


def predict_scenario(
    source: pd.DataFrame,
    temperature_increase: float,
    snowfall_reduction: int,
) -> pd.DataFrame:
    """Apply the scenario and run the saved preprocessing and classifier."""
    simulated = source.copy()
    if "t2m" in simulated:
        simulated["t2m"] = simulated["t2m"] + temperature_increase
    if "sf" in simulated:
        simulated["sf"] = simulated["sf"] * (1 - snowfall_reduction / 100)

    metadata_columns = [
        "glacier_name",
        "rgi_id",
        "country_left",
        "country_right",
        "region",
        "dem_source",
        "actual_mass_change",
    ]
    features = simulated.drop(
        columns=metadata_columns + list(collinear_dropped),
        errors="ignore",
    ).select_dtypes(include=[np.number])

    expected_features = getattr(scaler, "feature_names_in_", None)
    if expected_features is not None:
        missing_features = sorted(set(expected_features) - set(features.columns))
        if missing_features:
            raise ValueError(
                "The dataset is missing model features: " + ", ".join(missing_features)
            )
        features = features.loc[:, expected_features]
    features = features.fillna(0)

    scaled = scaler.transform(features)
    clusters = kmeans.predict(scaled)
    reduced = pca.transform(np.column_stack((scaled, clusters)))
    predictions = np.asarray(xgb_model.predict(reduced)).astype(int)

    simulated["Predicted_Status_2030"] = predictions
    simulated["Status_Label"] = simulated["Predicted_Status_2030"].map(
        {0: "Mass Loss (High Risk)", 1: "Mass Gain (Stable)"}
    ).fillna("Unknown")
    return simulated


if run_sim:
    with st.spinner("Running XGBoost 2030 simulation..."):
        try:
            result = predict_scenario(df, temp_increase, snow_decrease)
        except (ValueError, KeyError, RuntimeError) as error:
            st.error(f"Simulation failed: {error}")
            st.stop()

    total_glaciers = len(result)
    if total_glaciers == 0:
        st.warning("The prepared dataset contains no glacier records to simulate.")
        st.stop()

    loss_count = int((result["Predicted_Status_2030"] == 0).sum())
    gain_count = int((result["Predicted_Status_2030"] == 1).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Glaciers Analyzed", f"{total_glaciers:,}")
    col2.metric(
        "Glaciers Losing Mass (2030)",
        f"{loss_count:,}",
        f"{loss_count / total_glaciers * 100:.1f}%",
        delta_color="inverse",
    )
    col3.metric(
        "Glaciers Gaining Mass (2030)",
        f"{gain_count:,}",
        f"{gain_count / total_glaciers * 100:.1f}%",
    )

    st.markdown("---")
    map_columns = {"cenlat", "cenlon", "area_km2"}
    if map_columns.issubset(result.columns):
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.subheader("Spatial Risk Map")
            fig_map = px.scatter_map(
                result,
                lat="cenlat",
                lon="cenlon",
                color="Status_Label",
                size="area_km2",
                color_discrete_map={
                    "Mass Loss (High Risk)": "#ff4b4b",
                    "Mass Gain (Stable)": "#00cc96",
                },
                zoom=5.5,
                center={"lat": 28.3949, "lon": 84.1240},
                map_style="carto-darkmatter",
                title="Geographical Distribution of 2030 Glacier Status",
            )
            fig_map.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig_map, use_container_width=True)

        with fig_col2:
            st.subheader("Elevation vs. Mass Status")
            if "zmed_m" in result:
                fig_violin = px.violin(
                    result,
                    y="zmed_m",
                    x="Status_Label",
                    color="Status_Label",
                    color_discrete_map={
                        "Mass Loss (High Risk)": "#ff4b4b",
                        "Mass Gain (Stable)": "#00cc96",
                    },
                    box=True,
                    title="Median Elevation of Glaciers by Predicted Status (2030)",
                )
                fig_violin.update_layout(
                    yaxis_title="Median Elevation (m)",
                    xaxis_title="Predicted Status",
                )
                st.plotly_chart(fig_violin, use_container_width=True)
else:
    st.info(
        "👈 Adjust the Climate Simulation Controls in the sidebar and click "
        "**Run 2030 Simulation** to begin."
    )
