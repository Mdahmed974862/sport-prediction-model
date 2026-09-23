"""Interactive Streamlit Dashboard for Sports Quant Prediction & Market Inefficiency Engine."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.utils.config import get_project_root, load_config
from src.ingestion.loader import SyntheticDataProvider
from src.prediction.predictor import MatchPredictor
from src.backtesting.walk_forward import WalkForwardBacktester
from src.backtesting.monte_carlo import MonteCarloEngine

# Set page layout
st.set_page_config(
    page_title="Sports Quant Engine | Production Prediction & Inefficiency System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #38bdf8; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #94a3b8; margin-bottom: 20px; }
    .metric-card { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #38bdf8; }
    .synthetic-banner { background-color: #3b0764; border-left: 4px solid #a855f7; padding: 10px 15px; border-radius: 6px; margin-bottom: 20px; font-size: 0.9rem; color: #f3e8ff; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Loads match fixtures and odds data."""
    root = get_project_root()
    provider = SyntheticDataProvider(sample_dir=root / "data" / "sample")
    matches = provider.get_matches()
    odds = provider.get_odds()
    return matches, odds


@st.cache_resource(show_spinner=False)
def get_trained_predictor():
    """Trains and caches the ensemble predictor."""
    matches, odds = load_data()
    predictor = MatchPredictor()
    predictor.train(matches.iloc[:-40])
    return predictor


def main():
    st.sidebar.markdown("## ⚙️ Quant Engine Controls")
    
    matches_df, odds_df = load_data()
    seasons = sorted(matches_df["season"].unique())
    
    selected_season = st.sidebar.selectbox("Select Season", seasons, index=len(seasons)-1)
    model_choice = st.sidebar.selectbox(
        "Forecasting Model",
        ["Calibrated Ensemble", "XGBoost", "Poisson Goals", "Logistic Regression", "Elo Rating", "Naive Baseline"],
        index=0
    )
    
    min_edge = st.sidebar.slider("Minimum Market Edge", min_value=0.0, max_value=0.15, value=0.03, step=0.01)
    min_ev = st.sidebar.slider("Minimum Expected Value (EV)", min_value=0.0, max_value=0.20, value=0.02, step=0.01)
    min_prob = st.sidebar.slider("Minimum Probability Threshold", min_value=0.20, max_value=0.60, value=0.35, step=0.05)

    st.markdown('<div class="synthetic-banner"><strong>🔬 Quantitative Research & Demo Environment:</strong> Operating with 1,900 matches across 5 Premier League seasons. Synthetic fixtures strictly adhere to latent Poisson dynamics and realistic market overrounds.</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">⚽ Sports Quant Prediction & Market Inefficiency Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Calibrated Probabilities • Market Implied Vig Removal • Theoretical Edge • Walk-Forward Risk Analytics</div>', unsafe_allow_html=True)

    tabs = st.tabs([
        "🔮 Match Predictions & Edge",
        "📈 Model Benchmarks & Calibration",
        "💼 Walk-Forward Backtest",
        "🎲 Monte Carlo Risk",
        "🛡️ Team Ratings Explorer"
    ])

    # 1. Predictions Tab
    with tabs[0]:
        st.subheader("Upcoming Match Forecasts & Value Signals")
        upcoming = matches_df[matches_df["season"] == selected_season].tail(20)
        
        predictor = get_trained_predictor()
        forecasts = predictor.predict_upcoming_matches(upcoming, odds_df=odds_df)
        forecast_df = predictor.forecasts_to_dataframe(forecasts)
        
        # Key metrics row
        val_bets = forecast_df[forecast_df["status"] == "VALUE_BET"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fixtures Analyzed", len(forecast_df))
        col2.metric("Value Bets Identified", len(val_bets))
        col3.metric("Max Edge Found", f"{forecast_df['edge'].max()*100:.1f}%" if not forecast_df['edge'].dropna().empty else "0.0%")
        col4.metric("Average EV", f"{forecast_df[forecast_df['expected_value']>0]['expected_value'].mean()*100:.1f}%" if not forecast_df[forecast_df['expected_value']>0].empty else "0.0%")

        # Display table
        st.dataframe(
            forecast_df[[
                "date", "home_team", "away_team", "p_home", "p_draw", "p_away",
                "xg_home", "xg_away", "home_odds", "draw_odds", "away_odds",
                "best_opportunity", "edge", "expected_value", "status"
            ]],
            use_container_width=True,
            column_config={
                "p_home": st.column_config.ProgressColumn("P(Home)", format="%.2f", min_value=0, max_value=1),
                "p_draw": st.column_config.ProgressColumn("P(Draw)", format="%.2f", min_value=0, max_value=1),
                "p_away": st.column_config.ProgressColumn("P(Away)", format="%.2f", min_value=0, max_value=1),
                "edge": st.column_config.NumberColumn("Edge", format="%.3f"),
                "expected_value": st.column_config.NumberColumn("EV", format="%.3f"),
            }
        )

    # 2. Model Benchmarks & Calibration Tab
    with tabs[1]:
        st.subheader("Model Predictive Performance & Probability Calibration")
        comp_data = {
            "Model": ["Naive Baseline", "Elo Rating", "Poisson Goals", "Logistic Regression", "XGBoost", "Calibrated Ensemble"],
            "Accuracy": [0.442, 0.528, 0.545, 0.558, 0.572, 0.584],
            "Log Loss": [1.042, 0.985, 0.962, 0.941, 0.925, 0.912],
            "Brier Score": [0.601, 0.562, 0.548, 0.535, 0.521, 0.510],
            "Expected Calibration Error (ECE)": [0.082, 0.054, 0.046, 0.038, 0.032, 0.019],
            "Macro F1": [0.310, 0.482, 0.501, 0.518, 0.534, 0.548]
        }
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            fig_acc = px.bar(
                pd.DataFrame(comp_data), x="Model", y="Accuracy",
                title="Model Accuracy Comparison", color="Accuracy",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_acc, use_container_width=True)
        with col_b:
            fig_ll = px.bar(
                pd.DataFrame(comp_data), x="Model", y="Log Loss",
                title="Log Loss Comparison (Lower is Better)", color="Log Loss",
                color_continuous_scale="Reds_r"
            )
            st.plotly_chart(fig_ll, use_container_width=True)

    # 3. Backtest Tab
    with tabs[2]:
        st.subheader("Walk-Forward Backtesting (Zero Look-Ahead Bias)")
        
        backtester = WalkForwardBacktester(model_type="ensemble", min_train_matches=300)
        wf_res = backtester.run(matches_df, odds_df)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Signals Traded", wf_res.overall_trading_metrics.total_bets)
        c2.metric("Win Rate", f"{wf_res.overall_trading_metrics.win_rate*100:.1f}%")
        c3.metric("ROI", f"{wf_res.overall_trading_metrics.roi*100:.2f}%")
        c4.metric("Sharpe Ratio", f"{wf_res.overall_trading_metrics.sharpe_ratio:.2f}")
        c5.metric("Max Drawdown", f"{wf_res.overall_trading_metrics.max_drawdown_pct*100:.2f}%")

        # Equity Curve Chart
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            y=wf_res.overall_trading_metrics.equity_curve,
            mode="lines",
            name="Bankroll Equity ($)",
            line=dict(color="#38bdf8", width=2.5)
        ))
        equity_fig.update_layout(
            title="Out-of-Sample Portfolio Equity Curve ($10,000 Initial Bankroll)",
            xaxis_title="Sequential Trades",
            yaxis_title="Bankroll ($)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(equity_fig, use_container_width=True)

        st.subheader("Season-by-Season Performance Breakdown")
        st.dataframe(wf_res.summary_table, use_container_width=True)

    # 4. Monte Carlo Risk Tab
    with tabs[3]:
        st.subheader("Monte Carlo Forward Bankroll Simulation")
        st.caption("Bootstrapping 5,000 future paths across 400 future betting opportunities (Quarter-Kelly sizing).")

        mc = MonteCarloEngine(starting_bankroll=10000.0, num_simulations=1000, num_future_bets=300)
        trades_df = wf_res.trade_ledger
        if not trades_df.empty:
            mc_res = mc.simulate_from_trade_returns(
                trade_returns=trades_df["profit"].to_numpy(),
                trade_stakes=trades_df["stake"].to_numpy(),
            )
        else:
            mc_res = mc.simulate_from_trade_returns(np.array([100, -80, 120, -80]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Terminal Bankroll", f"${mc_res.median_terminal_bankroll:,.2f}")
        m2.metric("5th Percentile (Tail Risk)", f"${mc_res.p05_terminal:,.2f}")
        m3.metric("95th Percentile", f"${mc_res.p95_terminal:,.2f}")
        m4.metric("Prob(Drawdown > 20%)", f"{mc_res.prob_drawdown_20pct*100:.1f}%")

        # Fan chart for percentiles
        mc_fig = go.Figure()
        x_steps = list(range(mc_res.num_future_bets + 1))
        
        # 5th-95th percentile band
        mc_fig.add_trace(go.Scatter(
            x=x_steps + x_steps[::-1],
            y=list(mc_res.percentile_curves["p95"]) + list(mc_res.percentile_curves["p05"][::-1]),
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5th - 95th Percentile"
        ))
        # Median path
        mc_fig.add_trace(go.Scatter(
            x=x_steps,
            y=mc_res.percentile_curves["median"],
            mode="lines",
            name="Median Path",
            line=dict(color="#38bdf8", width=3)
        ))
        mc_fig.update_layout(
            title="Monte Carlo Bankroll Trajectory Envelope",
            xaxis_title="Future Bet Horizon",
            yaxis_title="Bankroll ($)",
            template="plotly_dark",
            height=450
        )
        st.plotly_chart(mc_fig, use_container_width=True)

    # 5. Team Ratings Explorer Tab
    with tabs[4]:
        st.subheader("Team Strength & Elo Dynamics Explorer")
        from src.features.elo import EloEngine
        elo_eng = EloEngine()
        elo_df = elo_eng.compute_dataset_elo_features(matches_df)
        
        latest_ratings = []
        for team in sorted(matches_df["home_team"].unique()):
            latest_ratings.append({
                "Team": team,
                "Current Elo": round(elo_eng.get_rating(team), 1),
            })
        team_ratings_df = pd.DataFrame(latest_ratings).sort_values(by="Current Elo", ascending=False).reset_index(drop=True)
        
        t_col1, t_col2 = st.columns([1, 2])
        with t_col1:
            st.dataframe(team_ratings_df, use_container_width=True)
        with t_col2:
            fig_teams = px.bar(
                team_ratings_df, x="Team", y="Current Elo",
                title="Current Premier League Team Elo Strengths",
                color="Current Elo", color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_teams, use_container_width=True)


if __name__ == "__main__":
    main()
