import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_player_data, load_team_data

port = int(os.environ.get("PORT", 8501)) 
st.run(port=port, address="0.0.0.0")

st.set_page_config(
    page_title="NBA Strategy Optimization",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #17408B;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #E03A3E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
    }
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .feature-description {
        font-size: 0.9rem;
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #E03A3E;
        margin: 10px 0;
    }
    .stat-title {
        font-size: 1rem;
        color: #17408B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("NBA Strategy Platform")
st.sidebar.markdown("Explore the platform features to analyze NBA data and optimize strategies.")

# Main content
st.markdown("<h1 class='main-header'>NBA Strategy Optimization Platform</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Advanced Analytics & AI for Basketball Strategy</p>", unsafe_allow_html=True)

# Introduction
st.markdown("""
This platform brings together data analysis, machine learning, and reinforcement learning
to provide comprehensive insights for basketball strategy optimization. Developed to support
coaches, analysts, and basketball enthusiasts, the system leverages real NBA data to deliver
actionable insights and strategic recommendations.
""")

# Quick league insights section
st.subheader("📊 Quick League Insights")

# Load data
@st.cache_data(ttl=3600)
def load_data(season='2022-23'):
    players_df = load_player_data(season)
    teams_df = load_team_data(season)
    return players_df, teams_df

players_df, teams_df = load_data()

# Check if data is loaded
if not players_df.empty:
    # Show some quick stats in a row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-title">Top Scorer</div>
                <div class="stat-value">{players_df.nlargest(1, 'PTS').iloc[0]['PLAYER_NAME']}</div>
                <div class="stat-title">{players_df.nlargest(1, 'PTS').iloc[0]['PTS']:.1f} PPG</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-title">Top Rebounder</div>
                <div class="stat-value">{players_df.nlargest(1, 'REB').iloc[0]['PLAYER_NAME']}</div>
                <div class="stat-title">{players_df.nlargest(1, 'REB').iloc[0]['REB']:.1f} RPG</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-title">Top Assists</div>
                <div class="stat-value">{players_df.nlargest(1, 'AST').iloc[0]['PLAYER_NAME']}</div>
                <div class="stat-title">{players_df.nlargest(1, 'AST').iloc[0]['AST']:.1f} APG</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="card">
                <div class="stat-title">Most Efficient</div>
                <div class="stat-value">{players_df[players_df['FGA'] > 500].nlargest(1, 'FG_PCT').iloc[0]['PLAYER_NAME']}</div>
                <div class="stat-title">{players_df[players_df['FGA'] > 500].nlargest(1, 'FG_PCT').iloc[0]['FG_PCT']*100:.1f}% FG</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    # Show a league overview visualization
    st.subheader("League Overview")
    tab1, tab2 = st.tabs(["Scoring Distribution", "Team Comparison"])
    
    with tab1:
        # Create a histogram of scoring distribution
        fig = px.histogram(
            players_df[players_df['MIN'] > 15],  # Filter for players with significant minutes
            x='PTS',
            nbins=20,
            title='Scoring Distribution Among NBA Players',
            labels={'PTS': 'Points Per Game', 'count': 'Number of Players'},
            color_discrete_sequence=['#17408B']
        )
        
        fig.update_layout(
            xaxis_title='Points Per Game',
            yaxis_title='Number of Players'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        if not teams_df.empty:
            # Compare team metrics
            metrics = st.selectbox(
                "Select metric for team comparison",
                ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
                format_func=lambda x: {
                    'PTS': 'Points Per Game',
                    'REB': 'Rebounds Per Game',
                    'AST': 'Assists Per Game',
                    'STL': 'Steals Per Game',
                    'BLK': 'Blocks Per Game',
                    'FG_PCT': 'Field Goal Percentage',
                    'FG3_PCT': '3-Point Percentage',
                    'FT_PCT': 'Free Throw Percentage'
                }.get(x, x)
            )
            
            # Create bar chart
            is_pct = metrics.endswith('_PCT')
            values = teams_df[metrics] * 100 if is_pct else teams_df[metrics]
            
            fig = px.bar(
                x=teams_df['TEAM_NAME'],
                y=values,
                title=f'Team Comparison: {metrics.replace("_", " ")}',
                labels={'y': 'Percentage' if is_pct else 'Per Game', 'x': 'Team'},
                color=values,
                color_continuous_scale='RdBu_r' if metrics in ['TOV', 'PF'] else 'BuRd_r',
                text=values.round(1)
            )
            
            fig.update_layout(
                xaxis={'categoryorder': 'total descending'},
                xaxis_tickangle=-45,
                yaxis_title='Percentage' if is_pct else 'Per Game'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Team data is not available. Please check your connection to the NBA API.")
else:
    st.warning("Player data could not be loaded. Please check your connection to the NBA API.")

# Features section
st.subheader("🔍 Platform Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Player Analysis</div>
            <div class="feature-description">
                Detailed player statistics and performance metrics. Compare players, track performance trends, and identify strengths and weaknesses.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">🔮</div>
            <div class="feature-title">Performance Prediction</div>
            <div class="feature-description">
                ML-powered predictions for player statistics and performance outcomes. Anticipate how players will perform in upcoming games.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Strategic Optimization</div>
            <div class="feature-description">
                RL-based decision optimization for game strategies. Learn optimal decision patterns for different game situations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Second row of features
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">👥</div>
            <div class="feature-title">Player Clustering</div>
            <div class="feature-description">
                Identify player types and playstyles using unsupervised learning. Group similar players to find hidden patterns.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">🏆</div>
            <div class="feature-title">Team Analysis</div>
            <div class="feature-description">
                Team-level metrics and matchup analysis. Understand team strengths, weaknesses, and optimal lineup compositions.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
            <div class="feature-icon">📱</div>
            <div class="feature-title">Real-time Updates</div>
            <div class="feature-description">
                Stay updated with the latest NBA statistics and trends. Data is refreshed regularly during the NBA season.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Getting started section
st.subheader("🚀 Getting Started")
st.markdown("""
1. **Explore Player Statistics:** Navigate to the Player Statistics page to explore individual player performance metrics.
2. **Predict Performance:** Use the Performance Prediction page to forecast player statistics using ML models.
3. **Optimize Strategy:** Check out the RL Decision Optimization page to see how AI can optimize in-game decisions.
4. **Analyze Teams:** Visit the Team Analysis page to compare teams and analyze matchups.
5. **Discover Player Types:** Explore the Player Clustering page to identify different player profiles and playstyles.

Use the sidebar to navigate between different features of the platform.
""")

# Footer
st.markdown("---")
st.markdown("Data sourced from NBA Stats API. Updated daily during the NBA season.")
st.markdown("© 2025 NBA Strategy Optimization Platform. All rights reserved.")
