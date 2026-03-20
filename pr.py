import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Simulación de Preferencias Reveladas FMCG", layout="wide")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SEGMENTS = ["Sensibles al precio", "Promo seekers", "Leales a marca", "Conveniencia"]
CHANNELS = ["Moderno", "Tradicional", "E-commerce"]
REGIONS = ["Norte", "Centro", "Sur"]
SEASONS = ["Baja", "Media", "Alta"]


@dataclass
class SimulationConfig:
    n_weeks: int
    n_markets: int
    n_skus: int
    outside_option_utility: float = 0.6


def softmax_by_group(df: pd.DataFrame, utility_col: str, group_cols: List[str]) -> pd.Series:
    temp = df.copy()
    max_u = temp.groupby(group_cols)[utility_col].transform("max")
    exp_u = np.exp(temp[utility_col] - max_u)
    sum_exp = exp_u.groupby([temp[c] for c in group_cols]).transform("sum")
    return exp_u / sum_exp


def clip01(x):
    return np.minimum(1.0, np.maximum(0.0, x))


def build_sku_master(n_skus: int) -> pd.DataFrame:
    brands = [f"Marca_{i+1}" for i in range(max(3, n_skus // 2))]
    categories = ["Bebidas", "Botanas", "Lácteos", "Limpieza", "Cuidado personal"]

    rows = []
    for i in range(n_skus):
        rows.append({
            "sku_id": f"SKU_{i+1}",
            "brand": np.random.choice(brands),
            "category": np.random.choice(categories),
            "base_price": round(np.random.uniform(15, 120), 2),
            "size_index": np.random.choice([0.8, 1.0, 1.2, 1.5]),
            "quality_index": round(np.random.uniform(0.7, 1.3), 3),
            "brand_equity": round(np.random.uniform(0.5, 1.5), 3),
            "margin_pct": round(np.random.uniform(0.18, 0.45), 3),
        })
    return pd.DataFrame(rows)


def build_true_segment_preferences() -> Dict[str, Dict[str, float]]:
    return {
        "Sensibles al precio": {
            "beta_price": -2.0,
            "beta_discount": 0.9,
            "beta_display": 0.3,
            "beta_distribution": 0.5,
            "beta_quality": 0.4,
            "beta_brand": 0.2,
            "beta_season_high": 0.1,
        },
        "Promo seekers": {
            "beta_price": -1.2,
            "beta_discount": 1.8,
            "beta_display": 0.7,
            "beta_distribution": 0.4,
            "beta_quality": 0.4,
            "beta_brand": 0.3,
            "beta_season_high": 0.15,
        },
        "Leales a marca": {
            "beta_price": -0.8,
            "beta_discount": 0.3,
            "beta_display": 0.1,
            "beta_distribution": 0.6,
            "beta_quality": 0.8,
            "beta_brand": 1.5,
            "beta_season_high": 0.25,
        },
        "Conveniencia": {
            "beta_price": -0.7,
            "beta_discount": 0.2,
            "beta_display": 0.2,
            "beta_distribution": 1.4,
            "beta_quality": 0.5,
            "beta_brand": 0.4,
            "beta_season_high": 0.2,
        },
    }


def generate_market_structure(cfg: SimulationConfig) -> pd.DataFrame:
    rows = []
    for i in range(cfg.n_markets):
        rows.append({
            "market_id": f"MKT_{i+1}",
            "region": np.random.choice(REGIONS),
            "channel": np.random.choice(CHANNELS),
        })
    return pd.DataFrame(rows)


def generate_panel_data(cfg: SimulationConfig, sku_master: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    weeks = np.arange(1, cfg.n_weeks + 1)
    rows = []

    for week in weeks:
        season = "Alta" if week % 13 in [11, 12, 0] else ("Media" if week % 13 in [5, 6, 7] else "Baja")
        season_multiplier = {"Baja": 0.97, "Media": 1.00, "Alta": 1.06}[season]

        for _, mkt in market_df.iterrows():
            channel_multiplier = {"Moderno": 1.00, "Tradicional": 1.05, "E-commerce": 0.96}[mkt["channel"]]

            for _, sku in sku_master.iterrows():
                raw_discount = np.random.choice([0, 0.05, 0.10, 0.15, 0.20], p=[0.45, 0.2, 0.18, 0.12, 0.05])
                display = np.random.binomial(1, 0.22)
                distribution = clip01(np.random.normal(0.86, 0.10))
                stockout = np.random.binomial(1, 0.06 if distribution > 0.75 else 0.12)
                active = 1 if stockout == 0 else 0
                noise = np.random.normal(1.0, 0.04)

                realized_price = sku["base_price"] * season_multiplier * channel_multiplier * (1 - raw_discount) * noise
                realized_price = max(5.0, realized_price)

                rows.append({
                    "week": week,
                    "season": season,
                    "market_id": mkt["market_id"],
                    "region": mkt["region"],
                    "channel": mkt["channel"],
                    "sku_id": sku["sku_id"],
                    "brand": sku["brand"],
                    "category": sku["category"],
                    "base_price": sku["base_price"],
                    "price": round(realized_price, 2),
                    "discount_pct": raw_discount,
                    "display": display,
                    "distribution": round(distribution, 3),
                    "stockout": stockout,
                    "active": active,
                    "size_index": sku["size_index"],
                    "quality_index": sku["quality_index"],
                    "brand_equity": sku["brand_equity"],
                    "margin_pct": sku["margin_pct"],
                })

    df = pd.DataFrame(rows)
    df["category_week_market_avg_price"] = df.groupby(["category", "week", "market_id"])["price"].transform("mean")
    df["relative_price"] = df["price"] / df["category_week_market_avg_price"]
    df["season_high"] = (df["season"] == "Alta").astype(int)
    df["season_medium"] = (df["season"] == "Media").astype(int)
    return df


def assign_segment_mix(market_df: pd.DataFrame) -> pd.DataFrame:
    mixes = []
    for _, row in market_df.iterrows():
        alpha = np.array([2.2, 1.8, 1.7, 1.5])
        if row["channel"] == "Tradicional":
            alpha += np.array([0.8, 0.3, 0.0, 0.0])
        elif row["channel"] == "E-commerce":
            alpha += np.array([0.0, 0.2, 0.2, 0.8])
        else:
            alpha += np.array([0.1, 0.5, 0.3, 0.2])

        p = np.random.dirichlet(alpha)
        mixes.append({
            "market_id": row["market_id"],
            "Sensibles al precio": p[0],
            "Promo seekers": p[1],
            "Leales a marca": p[2],
            "Conveniencia": p[3],
        })
    return pd.DataFrame(mixes)


def simulate_choice_from_true_preferences(
    panel_df: pd.DataFrame,
    market_mix_df: pd.DataFrame,
    true_prefs: Dict[str, Dict[str, float]],
    cfg: SimulationConfig
) -> pd.DataFrame:
    df = panel_df.merge(market_mix_df, on="market_id", how="left").copy()

    for seg in SEGMENTS:
        p = true_prefs[seg]
        util = (
            p["beta_price"] * np.log(df["relative_price"].clip(lower=0.5, upper=2.0))
            + p["beta_discount"] * df["discount_pct"]
            + p["beta_display"] * df["display"]
            + p["beta_distribution"] * df["distribution"]
            + p["beta_quality"] * df["quality_index"]
            + p["beta_brand"] * df["brand_equity"]
            + p["beta_season_high"] * df["season_high"]
        )
        util = util + np.where(df["active"] == 1, 0.0, -8.0)
        util = util + np.random.normal(0, 0.15, len(df))
        df[f"utility_{seg}"] = util

    group_cols = ["week", "market_id", "category"]

    for seg in SEGMENTS:
        df[f"prob_{seg}"] = softmax_by_group(df, f"utility_{seg}", group_cols)

    df["predicted_share"] = 0.0
    for seg in SEGMENTS:
        df["predicted_share"] += df[seg] * df[f"prob_{seg}"]

    df["inside_share"] = (1 - cfg.outside_option_utility / (cfg.outside_option_utility + 1.0)) * df["predicted_share"]

    cat_group = df.groupby(group_cols, as_index=False).agg(
        avg_distribution=("distribution", "mean"),
        mean_discount=("discount_pct", "mean"),
        season_high=("season_high", "max")
    )

    cat_group["base_category_demand"] = (
        500
        + 300 * cat_group["avg_distribution"]
        + 120 * cat_group["mean_discount"]
        + 90 * cat_group["season_high"]
        + np.random.normal(0, 35, len(cat_group))
    ).clip(lower=120)

    df = df.merge(cat_group[group_cols + ["base_category_demand"]], on=group_cols, how="left")
    df["units"] = (df["inside_share"] * df["base_category_demand"]).clip(lower=0)
    df["revenue"] = df["units"] * df["price"]
    df["gross_profit"] = df["revenue"] * df["margin_pct"]

    return df


def prepare_choice_dataset(sim_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["week", "market_id", "category"]
    df = sim_df.copy()

    df["share_observed"] = df.groupby(group_cols)["units"].transform(lambda x: x / (x.sum() + 1e-9))
    df["choice_rank"] = df.groupby(group_cols)["share_observed"].rank(ascending=False, method="first")
    df["chosen"] = (df["choice_rank"] == 1).astype(int)

    feature_cols = [
        "relative_price", "discount_pct", "display", "distribution",
        "quality_index", "brand_equity", "season_high", "season_medium"
    ]

    model_df = df[
        group_cols + ["sku_id", "brand", "channel", "region", "chosen", "share_observed", "price", "units", "revenue"] + feature_cols
    ].copy()

    model_df = pd.get_dummies(model_df, columns=["channel", "region", "brand"], drop_first=True)
    return model_df


def train_revealed_preference_model(model_df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    target = "chosen"
    ignore_cols = ["week", "market_id", "category", "sku_id", "chosen", "share_observed", "price", "units", "revenue"]
    feature_cols = [c for c in model_df.columns if c not in ignore_cols]

    X = model_df[feature_cols]
    y = model_df[target]

    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    pipe.fit(X, y)
    return pipe, feature_cols


def estimate_utilities_and_scores(model, feature_cols: List[str], model_df: pd.DataFrame) -> pd.DataFrame:
    scored = model_df.copy()
    scored["revealed_preference_score"] = model.predict_proba(scored[feature_cols])[:, 1]
    return scored


def run_scenario(
    base_df: pd.DataFrame,
    model,
    feature_cols: List[str],
    sku_target: str,
    price_change_pct: float,
    discount_change_pct: float,
    distribution_change: float,
    display_override: int
) -> pd.DataFrame:
    scen = base_df.copy()

    mask = scen["sku_id"] == sku_target
    scen.loc[mask, "price"] *= (1 + price_change_pct)
    scen.loc[mask, "discount_pct"] = clip01(scen.loc[mask, "discount_pct"] + discount_change_pct)
    scen.loc[mask, "distribution"] = clip01(scen.loc[mask, "distribution"] + distribution_change)
    scen.loc[mask, "display"] = display_override

    scen["category_week_market_avg_price"] = scen.groupby(["category", "week", "market_id"])["price"].transform("mean")
    scen["relative_price"] = scen["price"] / scen["category_week_market_avg_price"]

    model_scen = scen.copy()
    model_scen["share_observed"] = model_scen.groupby(["week", "market_id", "category"])["units"].transform(
        lambda x: x / (x.sum() + 1e-9)
    )

    model_scen = pd.get_dummies(
        model_scen[
            ["week", "market_id", "category", "sku_id", "brand", "channel", "region", "price", "units", "revenue",
             "relative_price", "discount_pct", "display", "distribution", "quality_index", "brand_equity",
             "season_high", "season_medium", "share_observed"]
        ],
        columns=["channel", "region", "brand"],
        drop_first=True
    )

    for col in feature_cols:
        if col not in model_scen.columns:
            model_scen[col] = 0

    model_scen["revealed_preference_score"] = model.predict_proba(model_scen[feature_cols])[:, 1]

    group_cols = ["week", "market_id", "category"]
    model_scen["scenario_share"] = model_scen.groupby(group_cols)["revealed_preference_score"].transform(
        lambda x: x / (x.sum() + 1e-9)
    )

    demand_base = scen.groupby(group_cols, as_index=False)["units"].sum().rename(columns={"units": "category_units_base"})
    model_scen = model_scen.merge(demand_base, on=group_cols, how="left")

    model_scen["scenario_units"] = model_scen["scenario_share"] * model_scen["category_units_base"]
    model_scen["scenario_revenue"] = model_scen["scenario_units"] * model_scen["price"]

    return model_scen


st.title("Simulación de preferencias reveladas a partir de Revenue Management en FMCG")
st.caption("Pricing, promoción, distribución y preferencia revelada por SKU.")

with st.sidebar:
    st.header("Configuración")
    data_mode = st.radio("Origen de datos", ["Simular datos", "Cargar CSV de ejemplo"])
    n_weeks = st.slider("Semanas", 8, 52, 26)
    n_markets = st.slider("Mercados", 3, 20, 8)
    n_skus = st.slider("SKUs", 6, 40, 16)
    regenerate = st.button("Regenerar")

if "sim_data_ready" not in st.session_state or regenerate:
    cfg = SimulationConfig(n_weeks=n_weeks, n_markets=n_markets, n_skus=n_skus)

    if data_mode == "Cargar CSV de ejemplo":
        uploaded = pd.read_csv("sample_data.csv")
        sim_df = uploaded.copy()
        sku_master = sim_df[["sku_id", "brand", "category", "base_price", "size_index", "quality_index", "brand_equity", "margin_pct"]].drop_duplicates()
        market_df = sim_df[["market_id", "region", "channel"]].drop_duplicates()
    else:
        sku_master = build_sku_master(cfg.n_skus)
        market_df = generate_market_structure(cfg)
        panel_df = generate_panel_data(cfg, sku_master, market_df)
        true_prefs = build_true_segment_preferences()
        market_mix_df = assign_segment_mix(market_df)
        sim_df = simulate_choice_from_true_preferences(panel_df, market_mix_df, true_prefs, cfg)

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

cfg = st.session_state["cfg"]
sku_master = st.session_state["sku_master"]
market_df = st.session_state["market_df"]
sim_df = st.session_state["sim_df"]
model_df = st.session_state["model_df"]
scored_df = st.session_state["scored_df"]
model = st.session_state["model"]
feature_cols = st.session_state["feature_cols"]

tab1, tab2, tab3, tab4 = st.tabs([
    "Resumen",
    "Preferencias reveladas",
    "Escenarios",
    "Datos y coeficientes"
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
        avg_price=("price", "mean")
    )

    fig = px.bar(
        sku_summary.sort_values("revenue", ascending=False),
        x="sku_id",
        y="revenue",
        color="category",
        hover_data=["brand", "units", "avg_price"],
        title="Revenue por SKU"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    pref_summary = scored_df.groupby("sku_id", as_index=False).agg(
        revealed_preference_score=("revealed_preference_score", "mean"),
        share_observed=("share_observed", "mean"),
        units=("units", "sum"),
        revenue=("revenue", "sum"),
        price=("price", "mean")
    ).sort_values("revealed_preference_score", ascending=False)

    fig = px.bar(
        pref_summary,
        x="sku_id",
        y="revealed_preference_score",
        hover_data=["share_observed", "units", "revenue", "price"],
        title="Score de preferencia revelada"
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
        sim_df.copy(), model, feature_cols, sku_target,
        price_change_pct, discount_change_pct, distribution_change, display_override
    )

    base_summary = sim_df.groupby("sku_id", as_index=False).agg(
        base_units=("units", "sum"),
        base_revenue=("revenue", "sum")
    )
    scen_summary = scenario_df.groupby("sku_id", as_index=False).agg(
        scenario_units=("scenario_units", "sum"),
        scenario_revenue=("scenario_revenue", "sum")
    )

    compare = base_summary.merge(scen_summary, on="sku_id", how="left")
    compare["delta_units"] = compare["scenario_units"] - compare["base_units"]
    compare["delta_revenue"] = compare["scenario_revenue"] - compare["base_revenue"]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Base", x=compare["sku_id"], y=compare["base_revenue"]))
    fig.add_trace(go.Bar(name="Escenario", x=compare["sku_id"], y=compare["scenario_revenue"]))
    fig.update_layout(barmode="group", title="Revenue base vs escenario")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(compare.sort_values("delta_revenue", ascending=False), use_container_width=True)

with tab4:
    st.dataframe(sku_master, use_container_width=True)

    clf = model.named_steps["clf"]
    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": clf.coef_[0]
    }).sort_values("coefficient", ascending=False)

    fig = px.bar(coef_df, x="feature", y="coefficient", title="Coeficientes del modelo")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(coef_df, use_container_width=True)
