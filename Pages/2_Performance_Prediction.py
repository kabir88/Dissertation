import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_player_data, get_player_info, get_player_headshot_url
from prediction_models import PlayerStatPredictor

st.set_page_config(
    page_title="Performance Prediction - NBA Strategy Optimization",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .prediction-header {
        font-size: 1.8rem;
        color: #17408B;
        margin-bottom: 10px;
    }
    .prediction-subheader {
        font-size: 1.2rem;
        color: #E03A3E;
        margin-bottom: 20px;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .predicted-value {
        font-size: 2rem;
        font-weight: bold;
        color: #E03A3E;
        text-align: center;
        margin: 10px 0;
    }
    .prediction-label {
        font-size: 1rem;
        font-weight: bold;
        color: #17408B;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Performance Prediction")
st.sidebar.markdown("Predict player performance using machine learning models.")

# Load data
@st.cache_data(ttl=3600)
def load_data(season='2022-23'):
    return load_player_data(season)

# Initialize prediction model
@st.cache_resource
def load_model():
    return PlayerStatPredictor()

players_df = load_data()
predictor = load_model()

# Main content
st.title("🔮 Performance Prediction")
st.markdown("Use machine learning to predict player performance and statistics.")

# Season selector
available_seasons = ["2022-23", "2021-22", "2020-21", "2019-20", "2018-19"]
selected_season = st.selectbox("Select Season for Training Data", available_seasons)

# Check if data is loaded
if players_df.empty:
    st.warning("Player data could not be loaded. Please check your connection to the NBA API.")
    st.stop()

# Tabs for different prediction features
tab1, tab2, tab3 = st.tabs(["Model Training", "Player Prediction", "Feature Importance"])

with tab1:
    st.header("Train Prediction Models")
    st.markdown("Train machine learning models to predict player statistics based on historical data.")
    
    # Model selection
    model_type = st.radio(
        "Select model type",
        ["Linear Regression", "Random Forest"],
        format_func=lambda x: x
    )
    
    model_key = "linear_reg" if model_type == "Linear Regression" else "random_forest"
    
    # Training parameters for Random Forest
    if model_type == "Random Forest":
        col1, col2 = st.columns(2)
        with col1:
            n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
        with col2:
            max_depth = st.slider("Maximum tree depth", 2, 20, 10, 1)
    
    # Training button
    if st.button("Train Model"):
        with st.spinner(f"Training {model_type} model..."):
            # Prepare data
            X_train, y_train = predictor.prepare_data(players_df)
            
            if X_train is None or y_train is None:
                st.error("Error preparing training data. Please check your dataset.")
            else:
                # Train model
                if model_type == "Linear Regression":
                    models = predictor.train_linear_regression()
                    st.success("Linear Regression model trained successfully!")
                else:
                    models = predictor.train_random_forest(n_estimators=n_estimators, max_depth=max_depth)
                    st.success(f"Random Forest model trained successfully with {n_estimators} trees!")
                
                # Show model evaluation
                eval_df = predictor.evaluate_models(model_type=model_key)
                
                if eval_df is not None:
                    st.subheader("Model Evaluation")
                    st.dataframe(eval_df.style.format({
                        'RMSE': '{:.2f}',
                        'R2': '{:.3f}'
                    }))
                    
                    # Visualize model performance for a stat
                    st.subheader("Model Visualization")
                    stat_to_viz = st.selectbox(
                        "Select statistic to visualize predictions vs actual",
                        predictor.targets if predictor.targets else ["PTS", "REB", "AST"]
                    )
                    
                    fig = predictor.plot_predictions_vs_actual(stat_to_viz, model_type=model_key)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No visualization available. Please check your model.")
                else:
                    st.info("No evaluation data available yet. Try running a prediction first.")

with tab2:
    st.header("Predict Player Performance")
    st.markdown("Use trained models to predict how a player will perform next season.")
    
    # Player selection
    all_players = sorted(players_df['PLAYER_NAME'].tolist())
    selected_player = st.selectbox("Select Player", all_players, key="prediction_player")
    
    # Get player data
    if selected_player:
        player_data = players_df[players_df['PLAYER_NAME'] == selected_player].iloc[0]
        player_id = player_data['PLAYER_ID']
        
        # Display player info
        col1, col2 = st.columns([1, 3])
        
        with col1:
            player_img_url = get_player_headshot_url(player_id)
            st.image(player_img_url, width=160)
        
        with col2:
            st.markdown(f"### {selected_player}")
            st.markdown(f"**Team:** {player_data['TEAM_ABBREVIATION']} | **Position:** {player_data['POSITION']} | **Age:** {player_data['AGE']}")
            st.markdown(f"**Current Stats:** {player_data['PTS']:.1f} PPG, {player_data['REB']:.1f} RPG, {player_data['AST']:.1f} APG")
        
        # Model selection for prediction
        prediction_model = st.radio(
            "Select model for prediction",
            ["Linear Regression", "Random Forest"],
            format_func=lambda x: x,
            key="prediction_model_type"
        )
        
        prediction_key = "linear_reg" if prediction_model == "Linear Regression" else "random_forest"
        
        # Allow adjustments to player attributes
        st.subheader("Adjust Player Attributes for Prediction")
        st.markdown("Modify player attributes to see how they affect predictions.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age_mod = st.number_input("Age Adjustment", -3, 5, 0, 1)
        with col2:
            min_mod = st.number_input("Minutes Adjustment", -10, 10, 0, 1)
        with col3:
            fga_mod = st.number_input("FG Attempts Adjustment", -5, 10, 0, 1)
        with col4:
            fg3a_mod = st.number_input("3PT Attempts Adjustment", -5, 10, 0, 1)
        
        # Create modified player data for prediction
        modified_player = player_data.copy()
        modified_player['AGE'] += age_mod
        modified_player['MIN'] += min_mod
        modified_player['FGA'] += fga_mod
        modified_player['FG3A'] += fg3a_mod
        
        # Prediction button
        if st.button("Predict Performance"):
            with st.spinner(f"Predicting {selected_player}'s performance..."):
                predictions = predictor.predict_stats(modified_player, model_type=prediction_key)
                
                if predictions:
                    st.markdown("<h3 class='prediction-header'>Predicted Performance</h3>", unsafe_allow_html=True)
                    
                    # Display predictions
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div class="prediction-label">Points</div>
                                <div class="predicted-value">{predictions['PTS']:.1f}</div>
                                <div class="prediction-label">PPG</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col2:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div class="prediction-label">Rebounds</div>
                                <div class="predicted-value">{predictions['REB']:.1f}</div>
                                <div class="prediction-label">RPG</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col3:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div class="prediction-label">Assists</div>
                                <div class="predicted-value">{predictions['AST']:.1f}</div>
                                <div class="prediction-label">APG</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    with col4:
                        st.markdown(
                            f"""
                            <div class="prediction-card">
                                <div class="prediction-label">Efficiency</div>
                                <div class="predicted-value">{predictions['FG_PCT']*100:.1f}%</div>
                                <div class="prediction-label">FG%</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                    
                    # Comparison with current stats
                    st.subheader("Performance Comparison")
                    
                    # Create DataFrame for comparison
                    comparison_df = pd.DataFrame({
                        'Statistic': ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
                        'Current': [player_data[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']],
                        'Predicted': [predictions[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]
                    })
                    
                    # Calculate difference and percent change
                    comparison_df['Difference'] = comparison_df['Predicted'] - comparison_df['Current']
                    comparison_df['Change %'] = (comparison_df['Difference'] / comparison_df['Current'] * 100).round(1)
                    
                    # Format percentages
                    for i, stat in enumerate(comparison_df['Statistic']):
                        if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                            comparison_df.loc[i, 'Current'] = f"{comparison_df.loc[i, 'Current']*100:.1f}%"
                            comparison_df.loc[i, 'Predicted'] = f"{comparison_df.loc[i, 'Predicted']*100:.1f}%"
                            comparison_df.loc[i, 'Difference'] = f"{comparison_df.loc[i, 'Difference']*100:.1f}%"
                    
                    # Display the comparison
                    st.dataframe(comparison_df)
                    
                    # Performance analysis text
                    st.subheader("Analysis")
                    
                    # Determine key changes
                    pts_change = predictions['PTS'] - player_data['PTS']
                    reb_change = predictions['REB'] - player_data['REB']
                    ast_change = predictions['AST'] - player_data['AST']
                    
                    analysis_text = f"Based on the {prediction_model} model, {selected_player} is predicted to "
                    
                    if pts_change > 1:
                        analysis_text += f"**increase scoring by {pts_change:.1f} points** per game. "
                    elif pts_change < -1:
                        analysis_text += f"**decrease scoring by {abs(pts_change):.1f} points** per game. "
                    else:
                        analysis_text += f"maintain similar scoring levels. "
                    
                    if reb_change > 0.5 or ast_change > 0.5:
                        analysis_text += "Other notable changes include "
                        if reb_change > 0.5:
                            analysis_text += f"an increase of {reb_change:.1f} rebounds "
                        if ast_change > 0.5:
                            analysis_text += f"{'and ' if reb_change > 0.5 else ''}an increase of {ast_change:.1f} assists "
                        analysis_text += "per game."
                    
                    st.markdown(analysis_text)
                    
                    # Factors affecting prediction
                    st.markdown("**Key Factors Affecting Prediction:**")
                    st.markdown(f"- Age adjustment: {age_mod:+d} years")
                    st.markdown(f"- Minutes adjustment: {min_mod:+d} minutes")
                    st.markdown(f"- Shot attempts adjustment: {fga_mod:+d} FGA, {fg3a_mod:+d} 3PA")
                else:
                    st.error("Prediction failed. Please train a model first or check your data.")

with tab3:
    st.header("Feature Importance Analysis")
    st.markdown("Understand which factors most influence player performance predictions.")
    
    # Only show feature importance for Random Forest
    if predictor.rf_models:
        # Get feature importance
        importance_dict = predictor.get_feature_importance(model_type='random_forest')
        
        if importance_dict:
            # Let user select which target to analyze
            target_stat = st.selectbox(
                "Select statistic to analyze feature importance",
                list(importance_dict.keys())
            )
            
            # Plot feature importance
            fig = predictor.plot_feature_importance(target_stat, model_type='random_forest')
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
                
                # Explanation of top features
                st.subheader("Interpretation")
                
                top_features = importance_dict[target_stat].head(3)['Feature'].tolist()
                
                st.markdown(f"""
                For predicting a player's **{target_stat}**, the most important factors are:
                
                1. **{top_features[0]}**: This is the strongest predictor of {target_stat}.
                2. **{top_features[1]}**: Second most important feature.
                3. **{top_features[2]}**: Third most important feature.
                
                Understanding these relationships can help in player development and strategy planning.
                """)
            else:
                st.info("Feature importance visualization is not available. Please check your model.")
        else:
            st.info("Feature importance data is not available. Please train a Random Forest model first.")
    else:
        st.info("Feature importance is only available for Random Forest models. Please train a Random Forest model first.")

# Explanation of models
st.sidebar.markdown("---")
st.sidebar.subheader("About the Models")
st.sidebar.markdown("""
**Linear Regression**: Predicts a linear relationship between player attributes and performance metrics.

**Random Forest**: An ensemble of decision trees that can capture non-linear relationships for more complex predictions.
""")

# Footer
st.markdown("---")
st.markdown("Data sourced from NBA Stats API. Models are trained on historical NBA data.")
st.markdown("Use the sidebar to navigate to other analysis tools.")