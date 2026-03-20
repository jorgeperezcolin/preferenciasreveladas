import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_generator import (
    SimulationConfig,
    build_sku_master,
    build_true_segment_preferences,
    generate_market_structure,
    generate_panel_data,
    assign_segment_mix,
    simulate_choice_from_true_preferences,
)
from src.preference_model import (
    prepare_choice_dataset,
    train_revealed_preference_model,
    estimate_utilities_and_scores,
)
from src.scenario_engine import run_scenario
from src.indifference import (
    build_indifference_grid,
    build_indifference_curves_from_levels,
    infer_segment_indifference_params,
    build_sku_indifference_points,
)

ROOT = pathlib.Path(__file__).resolve().parent
SAMPLE_DATA_PATH = ROOT / "sample_data.csv"

st.set_page_config(page_title="FMCG Revealed Preferences Simulator", layout="wide")

st.title("Simulación de preferencias reveladas en FMCG")
st.caption("Revenue management, preferencia revelada, escenarios y curvas de indiferencia.")


@st.cache_data(show_spinner=False)
def load_sample_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def build_simulated_dataset(n_weeks: int, n_markets: int, n_skus: int):
    cfg = SimulationConfig(n_weeks=n_weeks, n_markets=n_markets, n_skus=n_skus)
    sku_master = build_sku_master(cfg.n_skus)
    market_df = generate_market_structure(cfg)
    panel_df = generate_panel_data(cfg, sku_master, market_df)
    true_prefs = build_true_segment_preferences()
    market_mix_df = assign_segment_mix(market_df)
    sim_df = simulate_choice_from_true_preferences(panel_df, market_mix_df, true_prefs, cfg)
    return cfg, sku_master, market_df, sim_df


@st.cache_data(show_spinner=False)
def prepare_uploaded_dataset(uploaded_bytes: bytes):
    sim_df = pd.read_csv(pd.io.common.BytesIO(uploaded_bytes))
    required_cols = {
        "week", "season", "market_id", "region", "channel", "sku_id", "brand", "category",
        "base_price", "price", "discount_pct", "display", "distribution", "stockout", "active",
        "size_index", "quality_index", "brand_equity", "margin_pct"
    }
    missing = sorted(required_cols - set(sim_df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {', '.join(missing)}")

    if "relative_price" not in sim_df.columns:
        sim_df["category_week_market_avg_price"] = sim_df.groupby(["category", "week", "market_id"])["price"].transform("mean")
        sim_df["relative_price"] = sim_df["price"] / sim_df["category_week_market_avg_price"]
    if "season_high" not in sim_df.columns:
        sim_df["season_high"] = (sim_df["season"] == "Alta").astype(int)
    if "season_medium" not in sim_df.columns:
        sim_df["season_medium"] = (sim_df["season"] == "Media").astype(int)
    if "units" not in sim_df.columns:
        # Heurística mínima para permitir demo con CSV operativos incompletos
        sim_df["units"] = 100 * sim_df["distribution"] * (1 + sim_df["discount_pct"]).clip(lower=0.1)
    if "revenue" not in sim_df.columns:
        sim_df["revenue"] = sim_df["units"] * sim_df["price"]
    if "gross_profit" not in sim_df.columns:
        sim_df["gross_profit"] = sim_df["revenue"] * sim_df["margin_pct"]

    sku_master = sim_df[["sku_id", "brand", "category", "base_price", "size_index", "quality_index", "brand_equity", "margin_pct"]].drop_duplicates()
    market_df = sim_df[["market_id", "region", "channel"]].drop_duplicates()
    cfg = SimulationConfig(
        n_weeks=int(sim_df["week"].nunique()),
        n_markets=int(sim_df["market_id"].nunique()),
        n_skus=int(sim_df["sku_id"].nunique()),
    )
    return cfg, sku_master, market_df, sim_df


with st.sidebar:
    st.header("Configuración")
    data_mode = st.radio("Origen de datos", ["Simular datos", "CSV de ejemplo", "Subir CSV propio"])
    n_weeks = st.slider("Semanas", 8, 52, 26)
    n_markets = st.slider("Mercados", 3, 20, 8)
    n_skus = st.slider("SKUs", 6, 40, 16)
    uploaded_file = st.file_uploader("CSV propio", type=["csv"])
    regenerate = st.button("Regenerar")

need_refresh = (
    "sim_data_ready" not in st.session_state
    or regenerate
    or st.session_state.get("data_mode") != data_mode
)

try:
    if need_refresh:
        if data_mode == "CSV de ejemplo":
            if not SAMPLE_DATA_PATH.exists():
                raise FileNotFoundError("No existe sample_data.csv en la raíz del proyecto.")
            sample_df = load_sample_dataset(str(SAMPLE_DATA_PATH))
            cfg, sku_master, market_df, sim_df = prepare_uploaded_dataset(sample_df.to_csv(index=False).encode("utf-8"))
        elif data_mode == "Subir CSV propio":
            if uploaded_file is None:
                st.info("Sube un CSV para inicializar el modelo.")
                st.stop()
            cfg, sku_master, market_df, sim_df = prepare_uploaded_dataset(uploaded_file.getvalue())
        else:
            cfg, sku_master, market_df, sim_df = build_simulated_dataset(n_weeks, n_markets, n_skus)

        model_df = prepare_choice_dataset(sim_df)
        model, feature_cols = train_revealed_preference_model(model_df)
        scored_df = estimate_utilities_and_scores(model, feature_cols, model_df)

        st.session_state["cfg"] = cfg
        st.session_state["sku_master"] = sku_master
        st.session_state["market_df"] = market_df
        st.session_state["sim_df"] = sim_df
        st.session_state["model_df"] = model_df
        st.session_state["scored_df"] = scored_df
        st.session_state["model"] = model
        st.session_state["feature_cols"] = feature_cols
        st.session_state["sim_data_ready"] = True
        st.session_state["data_mode"] = data_mode
except Exception as exc:
    st.error(f"No se pudo preparar el dataset: {exc}")
    st.stop()

cfg = st.session_state["cfg"]
sku_master = st.session_state["sku_master"]
sim_df = st.session_state["sim_df"]
scored_df = st.session_state["scored_df"]
model = st.session_state["model"]
feature_cols = st.session_state["feature_cols"]

with st.expander("Resumen técnico del dataset", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.write(f"Semanas: **{cfg.n_weeks}**")
    c2.write(f"Mercados: **{cfg.n_markets}**")
    c3.write(f"SKUs: **{cfg.n_skus}**")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen",
    "Preferencias reveladas",
    "Escenarios",
    "Curvas de indiferencia",
    "Datos y coeficientes",
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Unidades totales", f"{sim_df['units'].sum():,.0f}")
    c2.metric("Revenue total", f"${sim_df['revenue'].sum():,.0f}")
    c3.metric("Gross profit", f"${sim_df['gross_profit'].sum():,.0f}")
    c4.metric("SKUs", f"{sim_df['sku_id'].nunique():,.0f}")

    sku_summary = sim_df.groupby(["sku_id", "brand", "category"], as_index=False).agg(
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        avg_price=("price", "mean"),
        avg_distribution=("distribution", "mean"),
    )

    fig = px.bar(
        sku_summary.sort_values("revenue", ascending=False),
        x="sku_id",
        y="revenue",
        color="category",
        hover_data=["brand", "units", "avg_price", "avg_distribution"],
        title="Revenue por SKU",
    )
    st.plotly_chart(fig, use_container_width=True)

    bubble = px.scatter(
        sku_summary,
        x="avg_price",
        y="units",
        size="revenue",
        color="category",
        hover_name="sku_id",
        title="Mapa precio-volumen",
    )
    st.plotly_chart(bubble, use_container_width=True)

with tab2:
    pref_summary = scored_df.groupby("sku_id", as_index=False).agg(
        revealed_preference_score=("revealed_preference_score", "mean"),
        share_observed=("share_observed", "mean"),
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        price=("price", "mean"),
    ).sort_values("revealed_preference_score", ascending=False)

    fig = px.bar(
        pref_summary,
        x="sku_id",
        y="revealed_preference_score",
        hover_data=["share_observed", "units", "revenue", "price"],
        title="Score de preferencia revelada",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pref_summary, use_container_width=True)

with tab3:
    sku_target = st.selectbox("SKU objetivo", sorted(sim_df["sku_id"].unique().tolist()))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        price_change_pct = st.slider("Cambio % precio", -0.30, 0.30, 0.00, 0.01)
    with c2:
        discount_change_pct = st.slider("Cambio descuento", -0.20, 0.20, 0.00, 0.01)
    with c3:
        distribution_change = st.slider("Cambio distribución", -0.30, 0.30, 0.00, 0.01)
    with c4:
        display_override = st.selectbox("Display", [0, 1], index=1)

    scenario_df = run_scenario(
        sim_df.copy(),
        model,
        feature_cols,
        sku_target,
        price_change_pct,
        discount_change_pct,
        distribution_change,
        display_override,
    )

    base_summary = sim_df.groupby("sku_id", as_index=False).agg(
        base_units=("units", "sum"),
        base_revenue=("revenue", "sum"),
    )
    scen_summary = scenario_df.groupby("sku_id", as_index=False).agg(
        scenario_units=("scenario_units", "sum"),
        scenario_revenue=("scenario_revenue", "sum"),
    )
    compare = base_summary.merge(scen_summary, on="sku_id", how="left")
    compare["delta_units"] = compare["scenario_units"] - compare["base_units"]
    compare["delta_revenue"] = compare["scenario_revenue"] - compare["base_revenue"]

    target_row = compare.loc[compare["sku_id"] == sku_target].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Revenue base", f"${target_row['base_revenue']:,.0f}")
    k2.metric("Revenue escenario", f"${target_row['scenario_revenue']:,.0f}", f"${target_row['delta_revenue']:,.0f}")
    k3.metric("Unidades base", f"{target_row['base_units']:,.0f}")
    k4.metric("Unidades escenario", f"{target_row['scenario_units']:,.0f}", f"{target_row['delta_units']:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=compare["sku_id"], y=compare["base_revenue"]))
    fig.add_trace(go.Bar(name="Escenario", x=compare["sku_id"], y=compare["scenario_revenue"]))
    fig.update_layout(barmode="group", title="Revenue base vs escenario")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(compare.sort_values("delta_revenue", ascending=False), use_container_width=True)

with tab4:
    st.subheader("Curvas de indiferencia")
    c1, c2, c3 = st.columns(3)
    with c1:
        segment_name = st.selectbox("Segmento", ["Sensibles al precio", "Promo seekers", "Leales a marca", "Conveniencia"])
    with c2:
        reference_price = st.number_input("Precio de referencia", min_value=1.0, value=50.0, step=1.0)
    with c3:
        utility_scale = st.selectbox("Niveles de utilidad", ["Baja", "Media", "Alta"], index=1)

    params = infer_segment_indifference_params(segment_name)
    alpha = params["alpha_price_value"]
    beta = params["beta_brand_value"]

    utility_levels_map = {
        "Baja": [0.7, 0.9, 1.1],
        "Media": [1.0, 1.3, 1.6],
        "Alta": [1.4, 1.8, 2.2],
    }
    curves_df = build_indifference_curves_from_levels(
        utility_levels=utility_levels_map[utility_scale],
        brand_values=np.linspace(0.5, 1.8, 140),
        reference_price=reference_price,
        alpha_price_value=alpha,
        beta_brand_value=beta,
    )
    sku_points = build_sku_indifference_points(sim_df)

    fig_curves = px.line(
        curves_df,
        x="brand_value",
        y="price",
        color="utility_level",
        title=f"Curvas de indiferencia — {segment_name}",
    )
    fig_points = px.scatter(sku_points, x="brand_value", y="price", size="revenue", hover_name="sku_id")
    for trace in fig_points.data:
        fig_curves.add_trace(trace)
    fig_curves.update_layout(xaxis_title="Valor de marca / calidad percibida", yaxis_title="Precio promedio")
    st.plotly_chart(fig_curves, use_container_width=True)

    grid_df = build_indifference_grid(
        price_min=max(5.0, float(sku_points["price"].min()) * 0.7),
        price_max=float(sku_points["price"].max()) * 1.3,
        reference_price=reference_price,
        alpha_price_value=alpha,
        beta_brand_value=beta,
        utility_type="cobb_douglas",
        n_points=100,
    )
    heatmap = px.density_heatmap(
        grid_df,
        x="brand_value",
        y="price",
        z="utility",
        histfunc="avg",
        nbinsx=35,
        nbinsy=35,
        title="Mapa de utilidad",
    )
    st.plotly_chart(heatmap, use_container_width=True)
    st.markdown(
        f"**Lectura:** para **{segment_name}**, el peso del valor económico del precio es **{alpha:.2f}** y el peso de marca/calidad es **{beta:.2f}**."
    )

with tab5:
    st.dataframe(sku_master, use_container_width=True)
    clf = model.named_steps["clf"]
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": clf.coef_[0]}).sort_values("coefficient", ascending=False)
    fig = px.bar(coef_df, x="feature", y="coefficient", title="Coeficientes del modelo")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(coef_df, use_container_width=True)
