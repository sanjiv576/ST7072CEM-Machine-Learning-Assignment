import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Himalayan Glacier 2030 Simulator", layout="wide", page_icon="🏔️")

# --- CUSTOM CSS FOR PREMIUM DESIGN ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
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
""", unsafe_allow_html=True)

# --- CACHE DATA & MODELS ---
@st.cache_resource
def load_artifacts():
    df = pd.read_parquet("../final_datasets/clean_datasets/1_data_preparation_dataset.parquet", engine="fastparquet")
    to_drop = joblib.load("../pkl/collinear_dropped.pkl")
    scaler = joblib.load("../pkl/scaler.pkl")
    kmeans = joblib.load("../pkl/kmeans_model.pkl")
    pca = joblib.load("../pkl/pca_model.pkl")
    model = joblib.load("../pkl/final_xgb_model.pkl")
    return df, to_drop, scaler, kmeans, pca, model

try:
    df, to_drop, scaler, kmeans, pca, xgb_model = load_artifacts()
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# --- UI: HERO SECTION ---
st.title("🏔️ Himalayan Glacier 2030 Simulator")
st.markdown("**A Machine Learning Early-Warning System for Climate Hazards in Nepal**")
st.markdown("---")

# --- UI: SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Climate Simulation Controls")
st.sidebar.markdown("Adjust the meteorological drivers to simulate the year 2030.")

temp_increase = st.sidebar.slider(
    "Temperature Increase (°C)",
    min_value=0.0,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Simulate global warming by increasing the 2m surface temperature across all glaciers."
)

snow_decrease = st.sidebar.slider(
    "Snowfall Reduction (%)",
    min_value=0,
    max_value=50,
    value=10,
    step=5,
    help="Simulate reduced monsoonal snowfall."
)

st.sidebar.markdown("---")
run_sim = st.sidebar.button("Run 2030 Simulation")

# --- SIMULATION LOGIC ---
if run_sim:
    with st.spinner("Running XGBoost 2030 Simulation..."):
        # 1. Manipulate Data
        df_sim = df.copy()
        if 't2m' in df_sim.columns:
            df_sim['t2m'] += temp_increase
        if 'sf' in df_sim.columns:
            df_sim['sf'] = df_sim['sf'] * (1 - (snow_decrease/100))

        # 2. Preprocess
        raw_drop = ['glacier_name', 'rgi_id', 'country_left', 'country_right', 'region', 'dem_source', 'actual_mass_change']
        X = df_sim.drop(columns=raw_drop + to_drop, errors="ignore")
        X_num = X.select_dtypes(include=[np.number]).fillna(0)

        # 3. Pipeline transformations
        X_scaled = scaler.transform(X_num)
        clusters = kmeans.predict(X_scaled)
        X_scaled_with_cluster = np.column_stack((X_scaled, clusters))
        X_pca = pca.transform(X_scaled_with_cluster)

        # 4. Predict
        preds = xgb_model.predict(X_pca)
        df_sim['Predicted_Status_2030'] = preds
        df_sim['Status_Label'] = df_sim['Predicted_Status_2030'].map({0: 'Mass Loss (High Risk)', 1: 'Mass Gain (Stable)'})

        # --- DASHBOARD METRICS ---
        total_glaciers = len(preds)
        loss_count = sum(preds == 0)
        gain_count = sum(preds == 1)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Glaciers Analyzed", f"{total_glaciers:,}")
        col2.metric("Glaciers Losing Mass (2030)", f"{loss_count:,}", f"{((loss_count/total_glaciers)*100):.1f}%", delta_color="inverse")
        col3.metric("Glaciers Gaining Mass (2030)", f"{gain_count:,}", f"{((gain_count/total_glaciers)*100):.1f}%", delta_color="normal")

        st.markdown("---")

        # --- VISUALIZATIONS ---
        fig_col1, fig_col2 = st.columns([1, 1])

        with fig_col1:
            st.subheader("Spatial Risk Map")
            # Plotly Mapbox for spatial visualization
            fig_map = px.scatter_map(
                df_sim, 
                lat="cenlat", 
                lon="cenlon", 
                color="Status_Label",
                color_discrete_map={'Mass Loss (High Risk)': '#ff4b4b', 'Mass Gain (Stable)': '#00cc96'},
                size="area_km2",

                zoom=5.5,
                center={"lat": 28.3949, "lon": 84.1240},
                map_style="carto-darkmatter",
                title="Geographical Distribution of 2030 Glacier Status"
            )
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)

        with fig_col2:
            st.subheader("Elevation vs. Mass Status")
            # Violin plot showing elevation distribution of dying vs surviving glaciers
            fig_violin = px.violin(
                df_sim, 
                y="zmed_m", 
                x="Status_Label", 
                color="Status_Label",
                color_discrete_map={'Mass Loss (High Risk)': '#ff4b4b', 'Mass Gain (Stable)': '#00cc96'},
                box=True, 
                title="Median Elevation of Glaciers by Predicted Status (2030)"
            )
            fig_violin.update_layout(yaxis_title="Median Elevation (m)", xaxis_title="Predicted Status")
            st.plotly_chart(fig_violin, use_container_width=True)
else:
    st.info("👈 Adjust the Climate Simulation Controls in the sidebar and click **Run 2030 Simulation** to begin.")
