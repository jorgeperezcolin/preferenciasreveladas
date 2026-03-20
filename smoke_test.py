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


def main():
    cfg = SimulationConfig(n_weeks=10, n_markets=5, n_skus=12)
    sku_master = build_sku_master(cfg.n_skus)
    market_df = generate_market_structure(cfg)
    panel_df = generate_panel_data(cfg, sku_master, market_df)
    prefs = build_true_segment_preferences()
    market_mix_df = assign_segment_mix(market_df)
    sim_df = simulate_choice_from_true_preferences(panel_df, market_mix_df, prefs, cfg)

    assert not sim_df.empty
    assert sim_df["units"].sum() > 0
    assert sim_df["revenue"].sum() > 0

    model_df = prepare_choice_dataset(sim_df)
    model, feature_cols = train_revealed_preference_model(model_df)
    scored_df = estimate_utilities_and_scores(model, feature_cols, model_df)
    assert "revealed_preference_score" in scored_df.columns

    sku_target = sim_df["sku_id"].iloc[0]
    scenario_df = run_scenario(sim_df.copy(), model, feature_cols, sku_target, 0.05, 0.02, 0.03, 1)
    assert scenario_df["scenario_units"].sum() > 0
    assert scenario_df["scenario_revenue"].sum() > 0

    params = infer_segment_indifference_params("Leales a marca")
    curves_df = build_indifference_curves_from_levels([1.0, 1.3, 1.6], brand_values=__import__("numpy").linspace(0.5, 1.8, 100), reference_price=50.0, alpha_price_value=params["alpha_price_value"], beta_brand_value=params["beta_brand_value"])
    grid_df = build_indifference_grid(10, 120, reference_price=50.0, alpha_price_value=params["alpha_price_value"], beta_brand_value=params["beta_brand_value"], n_points=50)
    sku_points = build_sku_indifference_points(sim_df)

    assert not curves_df.empty
    assert not grid_df.empty
    assert not sku_points.empty

    print("Smoke test OK")
    print(f"Sim rows: {len(sim_df)}")
    print(f"Model rows: {len(model_df)}")
    print(f"Scenario rows: {len(scenario_df)}")
    print(f"Total units: {sim_df['units'].sum():,.2f}")
    print(f"Total revenue: {sim_df['revenue'].sum():,.2f}")
    print(f"Top SKU by revenue: {sim_df.groupby('sku_id')['revenue'].sum().sort_values(ascending=False).index[0]}")
    print(f"Indifference curve points: {len(curves_df)}")
    print(f"SKU indifference points: {len(sku_points)}")
    print(f"Model features: {len(feature_cols)}")


if __name__ == "__main__":
    main()
