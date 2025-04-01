import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_player_data, load_team_data, get_player_info, get_player_headshot_url, get_player_game_logs

st.set_page_config(
    page_title="Player Statistics - NBA Strategy Optimization",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .player-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 20px;
    }
    .player-img {
        width: 120px;
        height: 90px;
        object-fit: contain;
        border-radius: 5px;
    }
    .player-details {
        flex: 1;
    }
    .stat-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        height: 100%;
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
    .trend-value {
        font-size: 0.9rem;
    }
    .positive-trend {
        color: green;
    }
    .negative-trend {
        color: red;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Player Statistics")
st.sidebar.markdown("Analyze detailed player performance metrics and trends.")

# Load data on first run
@st.cache_data(ttl=3600)
def load_data(season='2022-23'):
    players_df = load_player_data(season)
    teams_df = load_team_data(season)
    return players_df, teams_df

players_df, teams_df = load_data()

# Top section
st.title("🏀 NBA Player Statistics")
st.markdown("Explore comprehensive statistics and performance metrics for NBA players.")

# Season selector
available_seasons = ["2022-23", "2021-22", "2020-21", "2019-20", "2018-19"]
selected_season = st.selectbox("Select Season", available_seasons)

# Check if data is loaded
if players_df.empty:
    st.warning("Player data could not be loaded. Please check your connection to the NBA API.")
    st.stop()

# Team filter
all_teams = sorted(players_df['TEAM_ABBREVIATION'].unique().tolist())
selected_team = st.selectbox("Filter by Team", ["All Teams"] + all_teams)

# Apply team filter
if selected_team != "All Teams":
    filtered_players = players_df[players_df['TEAM_ABBREVIATION'] == selected_team]
else:
    filtered_players = players_df

# Player selector
player_options = sorted(filtered_players['PLAYER_NAME'].tolist())
selected_player = st.selectbox("Select Player", player_options)

# Player analysis section
player_data = filtered_players[filtered_players['PLAYER_NAME'] == selected_player].iloc[0]
player_id = player_data['PLAYER_ID']

# Player header
col1, col2 = st.columns([1, 3])

with col1:
    player_img_url = get_player_headshot_url(player_id)
    st.image(player_img_url, width=160)

with col2:
    st.header(f"{selected_player}")
    st.subheader(f"{player_data['TEAM_ABBREVIATION']} | #{player_data.get('JERSEY_NUMBER', 'N/A')} | {player_data['POSITION']}")
    st.markdown(f"**Age:** {player_data['AGE']} | **Height:** {player_data.get('HEIGHT', 'N/A')} | **Weight:** {player_data.get('WEIGHT', 'N/A')} lbs")

# Key stats summary
st.subheader("Key Statistics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">Points</div>
            <div class="stat-value">{player_data['PTS']:.1f}</div>
            <div class="stat-title">PPG</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">Rebounds</div>
            <div class="stat-value">{player_data['REB']:.1f}</div>
            <div class="stat-title">RPG</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">Assists</div>
            <div class="stat-value">{player_data['AST']:.1f}</div>
            <div class="stat-title">APG</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">Field Goal</div>
            <div class="stat-value">{player_data['FG_PCT']*100:.1f}%</div>
            <div class="stat-title">FG%</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-title">3-Point</div>
            <div class="stat-value">{player_data['FG3_PCT']*100:.1f}%</div>
            <div class="stat-title">3P%</div>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Detailed statistics
st.subheader("Statistical Profile")
tab1, tab2, tab3 = st.tabs(["Overview", "Game Logs", "Shot Distribution"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Offensive Statistics")
        
        offensive_stats = {
            'Points Per Game': player_data['PTS'],
            'Field Goal %': player_data['FG_PCT'] * 100,
            '3-Point %': player_data['FG3_PCT'] * 100,
            'Free Throw %': player_data['FT_PCT'] * 100,
            'Offensive Rebounds': player_data['OREB'],
            'Assists': player_data['AST'],
            'Turnovers': player_data['TOV']
        }
        
        # Create DataFrame for chart
        off_df = pd.DataFrame({
            'Statistic': list(offensive_stats.keys()),
            'Value': list(offensive_stats.values())
        })
        
        # Calculate league averages for comparison
        league_avg = {
            'Points Per Game': players_df['PTS'].mean(),
            'Field Goal %': players_df['FG_PCT'].mean() * 100,
            '3-Point %': players_df['FG3_PCT'].mean() * 100,
            'Free Throw %': players_df['FT_PCT'].mean() * 100,
            'Offensive Rebounds': players_df['OREB'].mean(),
            'Assists': players_df['AST'].mean(),
            'Turnovers': players_df['TOV'].mean()
        }
        
        # Add comparison to league average
        off_df['League Average'] = off_df['Statistic'].map(league_avg)
        
        # Calculate percentile ranks (higher is better, except for turnovers)
        percentiles = {}
        for stat in offensive_stats:
            if stat == 'Turnovers':
                # For turnovers, lower is better
                percentile = (players_df['TOV'] >= player_data['TOV']).mean() * 100
            else:
                col_name = next((col for col in players_df.columns if col in stat.upper()), None)
                if col_name:
                    percentile = (players_df[col_name] <= player_data[col_name]).mean() * 100
                else:
                    percentile = 50  # Default if not found
            percentiles[stat] = percentile
        
        off_df['Percentile'] = off_df['Statistic'].map(percentiles)
        
        # Create the chart
        fig = px.bar(
            off_df,
            x='Statistic',
            y='Value',
            title='Offensive Statistics',
            text='Value',
            color='Percentile',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        # Add league average line
        fig.add_trace(
            go.Scatter(
                x=off_df['Statistic'],
                y=off_df['League Average'],
                mode='lines+markers',
                name='League Average',
                line=dict(color='rgba(0,0,0,0.5)', dash='dash')
            )
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### Defensive Statistics")
        
        defensive_stats = {
            'Defensive Rebounds': player_data['DREB'],
            'Steals': player_data['STL'],
            'Blocks': player_data['BLK'],
            'Personal Fouls': player_data['PF'],
            'Defensive Rating': player_data.get('DEF_RATING', 100)
        }
        
        # Create DataFrame for chart
        def_df = pd.DataFrame({
            'Statistic': list(defensive_stats.keys()),
            'Value': list(defensive_stats.values())
        })
        
        # Calculate league averages for comparison
        league_def_avg = {
            'Defensive Rebounds': players_df['DREB'].mean(),
            'Steals': players_df['STL'].mean(),
            'Blocks': players_df['BLK'].mean(),
            'Personal Fouls': players_df['PF'].mean(),
            'Defensive Rating': 110  # Example value
        }
        
        # Add comparison to league average
        def_df['League Average'] = def_df['Statistic'].map(league_def_avg)
        
        # Calculate percentile ranks (higher is better, except for personal fouls and def rating)
        def_percentiles = {}
        for stat in defensive_stats:
            if stat in ['Personal Fouls', 'Defensive Rating']:
                # For these stats, lower is better
                col_name = 'PF' if stat == 'Personal Fouls' else 'DEF_RATING'
                percentile = (players_df[col_name] >= player_data[col_name]).mean() * 100 if col_name in player_data else 50
            else:
                col_name = 'DREB' if stat == 'Defensive Rebounds' else 'STL' if stat == 'Steals' else 'BLK'
                percentile = (players_df[col_name] <= player_data[col_name]).mean() * 100
            def_percentiles[stat] = percentile
        
        def_df['Percentile'] = def_df['Statistic'].map(def_percentiles)
        
        # Create the chart
        fig = px.bar(
            def_df,
            x='Statistic',
            y='Value',
            title='Defensive Statistics',
            text='Value',
            color='Percentile',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        # Add league average line
        fig.add_trace(
            go.Scatter(
                x=def_df['Statistic'],
                y=def_df['League Average'],
                mode='lines+markers',
                name='League Average',
                line=dict(color='rgba(0,0,0,0.5)', dash='dash')
            )
        )
        
        fig.update_layout(
            xaxis_title='',
            yaxis_title='Value',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Game logs
    st.markdown("### Game-by-Game Statistics")
    with st.spinner("Loading game logs..."):
        game_logs = get_player_game_logs(player_id, season=selected_season)
        
        if not game_logs.empty:
            # Create a summary of recent games
            st.markdown("#### Recent Games")
            recent_games = game_logs.head(5)
            
            # Format the data for display
            display_logs = recent_games[['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']].copy()
            display_logs['GAME_DATE'] = pd.to_datetime(display_logs['GAME_DATE']).dt.strftime('%b %d, %Y')
            
            # Apply styling for wins and losses
            def highlight_wl(val):
                color = 'green' if val == 'W' else 'red' if val == 'L' else ''
                return f'background-color: {color}; color: white;' if color else ''
            
            styled_logs = display_logs.style.applymap(highlight_wl, subset=['WL'])
            
            # Display table
            st.dataframe(styled_logs, use_container_width=True)
            
            # Plot trends
            st.markdown("#### Performance Trends")
            
            # Allow user to select stats to plot
            stat_options = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
            selected_stats = st.multiselect("Select statistics to plot", stat_options, default=['PTS'])
            
            if selected_stats:
                # Create trend plot
                fig = go.Figure()
                
                for stat in selected_stats:
                    fig.add_trace(
                        go.Scatter(
                            x=game_logs['GAME_DATE'],
                            y=game_logs[stat],
                            mode='lines+markers',
                            name=stat
                        )
                    )
                    
                    # Add rolling average
                    if len(game_logs) > 5:
                        rolling_avg = game_logs[stat].rolling(window=5).mean()
                        fig.add_trace(
                            go.Scatter(
                                x=game_logs['GAME_DATE'],
                                y=rolling_avg,
                                mode='lines',
                                line=dict(dash='dash'),
                                name=f'{stat} (5-game avg)'
                            )
                        )
                
                fig.update_layout(
                    title='Game-by-Game Statistics',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Additional analysis of game logs
            if 'PTS' in game_logs.columns:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_pts_game = game_logs.loc[game_logs['PTS'].idxmax()]
                    st.markdown(f"**Best Scoring Game:** {max_pts_game['PTS']} pts vs {max_pts_game['MATCHUP'].split()[-1]} on {pd.to_datetime(max_pts_game['GAME_DATE']).strftime('%b %d')}")
                
                with col2:
                    max_reb_game = game_logs.loc[game_logs['REB'].idxmax()]
                    st.markdown(f"**Best Rebounding Game:** {max_reb_game['REB']} reb vs {max_reb_game['MATCHUP'].split()[-1]} on {pd.to_datetime(max_reb_game['GAME_DATE']).strftime('%b %d')}")
                
                with col3:
                    max_ast_game = game_logs.loc[game_logs['AST'].idxmax()]
                    st.markdown(f"**Best Assist Game:** {max_ast_game['AST']} ast vs {max_ast_game['MATCHUP'].split()[-1]} on {pd.to_datetime(max_ast_game['GAME_DATE']).strftime('%b %d')}")
        else:
            st.info("No game logs available for this player in the selected season.")

with tab3:
    # Shot distribution
    st.markdown("### Shot Distribution Analysis")
    
    # Field goal distribution
    col1, col2 = st.columns(2)
    
    with col1:
        # Calculate distribution of points
        two_pts = player_data['FG2M'] * 2 if 'FG2M' in player_data else (player_data['FGM'] - player_data['FG3M']) * 2
        three_pts = player_data['FG3M'] * 3
        ft_pts = player_data['FTM']
        
        total_pts = two_pts + three_pts + ft_pts
        
        # Create a pie chart of point sources
        fig = go.Figure(data=[go.Pie(
            labels=['2-Pointers', '3-Pointers', 'Free Throws'],
            values=[two_pts, three_pts, ft_pts],
            hole=0.4,
            marker_colors=['#E03A3E', '#17408B', '#C4CED4']
        )])
        
        fig.update_layout(
            title='Points Distribution',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Calculate shooting efficiency
        fg_pct = player_data['FG_PCT'] * 100
        fg3_pct = player_data['FG3_PCT'] * 100
        ft_pct = player_data['FT_PCT'] * 100
        efg_pct = (player_data['FGM'] + 0.5 * player_data['FG3M']) / player_data['FGA'] * 100 if player_data['FGA'] > 0 else 0
        ts_pct = (player_data['PTS'] / (2 * (player_data['FGA'] + 0.44 * player_data['FTA']))) * 100 if (player_data['FGA'] + 0.44 * player_data['FTA']) > 0 else 0
        
        # Create efficiency comparison chart
        efficiency_data = pd.DataFrame({
            'Category': ['Field Goal %', '3-Point %', 'Free Throw %', 'Effective FG %', 'True Shooting %'],
            'Percentage': [fg_pct, fg3_pct, ft_pct, efg_pct, ts_pct]
        })
        
        fig = px.bar(
            efficiency_data, 
            x='Category', 
            y='Percentage',
            color='Percentage',
            text_auto='.1f',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100]
        )
        
        fig.update_layout(
            title='Shooting Efficiency',
            xaxis_title='',
            yaxis_title='Percentage',
            yaxis_range=[0, 100],
            coloraxis_showscale=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional shot analysis text
    st.markdown("#### Shooting Analysis")
    
    # Calculate how many points come from each area
    pct_2pt = two_pts / total_pts * 100 if total_pts > 0 else 0
    pct_3pt = three_pts / total_pts * 100 if total_pts > 0 else 0
    pct_ft = ft_pts / total_pts * 100 if total_pts > 0 else 0
    
    st.markdown(f"""
    {selected_player}'s scoring profile shows:
    - **{pct_2pt:.1f}%** of points from 2-pointers 
    - **{pct_3pt:.1f}%** of points from 3-pointers
    - **{pct_ft:.1f}%** of points from free throws
    
    Overall efficiency rating (True Shooting): **{ts_pct:.1f}%**
    """)
    
    # Comparison to league average
    league_ts_pct = ((players_df['PTS'] / (2 * (players_df['FGA'] + 0.44 * players_df['FTA']))).mean() * 100) if players_df.shape[0] > 0 else 0
    
    if ts_pct > league_ts_pct:
        st.markdown(f"**Efficiency Assessment:** {selected_player} is **more efficient** than the league average ({league_ts_pct:.1f}%) in terms of True Shooting Percentage.")
    else:
        st.markdown(f"**Efficiency Assessment:** {selected_player} is **less efficient** than the league average ({league_ts_pct:.1f}%) in terms of True Shooting Percentage.")

# Player comparison section (placeholder)
if st.checkbox("Compare with another player"):
    st.subheader("Player Comparison")
    
    # Second player selector
    compare_player = st.selectbox("Select player to compare with", 
                                  [p for p in player_options if p != selected_player])
    
    if compare_player:
        compare_data = filtered_players[filtered_players['PLAYER_NAME'] == compare_player].iloc[0]
        
        # Create comparison dataframe
        compare_df = pd.DataFrame({
            'Statistic': ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT'],
            selected_player: [player_data[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']],
            compare_player: [compare_data[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']]
        })
        
        # Create a readable version of the stats with proper names
        stat_names = {
            'PTS': 'Points',
            'REB': 'Rebounds',
            'AST': 'Assists',
            'STL': 'Steals',
            'BLK': 'Blocks',
            'TOV': 'Turnovers',
            'FG_PCT': 'FG%',
            'FG3_PCT': '3PT%',
            'FT_PCT': 'FT%'
        }
        
        compare_df['Statistic'] = compare_df['Statistic'].map(stat_names)
        
        # Format percentage values
        for col in [selected_player, compare_player]:
            for i, stat in enumerate(compare_df['Statistic']):
                if stat in ['FG%', '3PT%', 'FT%']:
                    compare_df.loc[i, col] = f"{compare_df.loc[i, col]*100:.1f}%"
        
        # Display comparison
        st.dataframe(compare_df, use_container_width=True)
        
        # Radar chart for visual comparison
        categories = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
        
        # Normalize values for radar chart (0-1 scale)
        max_values = players_df[['PTS', 'REB', 'AST', 'STL', 'BLK']].max()
        
        player1_values = [player_data[stat] / max_values[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']]
        player2_values = [compare_data[stat] / max_values[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=player1_values + [player1_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=selected_player
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=player2_values + [player2_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=compare_player
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Player Comparison Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Textual analysis of comparison
        st.markdown("### Comparison Analysis")
        
        # Compare points
        pts_diff = player_data['PTS'] - compare_data['PTS']
        pts_analysis = f"{selected_player} scores **{abs(pts_diff):.1f} more points** per game." if pts_diff > 0 else f"{compare_player} scores **{abs(pts_diff):.1f} more points** per game."
        
        # Compare efficiency
        fg_diff = player_data['FG_PCT'] - compare_data['FG_PCT']
        fg_analysis = f"{selected_player} is **{abs(fg_diff)*100:.1f}% more efficient** in field goal percentage." if fg_diff > 0 else f"{compare_player} is **{abs(fg_diff)*100:.1f}% more efficient** in field goal percentage."
        
        # Overall assessment
        stat_wins = sum([1 for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK'] if player_data[stat] > compare_data[stat]])
        if stat_wins >= 3:
            overall = f"{selected_player} performs better in {stat_wins}/5 key statistical categories."
        else:
            overall = f"{compare_player} performs better in {5-stat_wins}/5 key statistical categories."
        
        st.markdown(f"""
        **Key Differences:**
        - {pts_analysis}
        - {fg_analysis}
        - **Overall:** {overall}
        """)

# Footer
st.markdown("---")
st.markdown("Data sourced from NBA Stats API. Updated daily during the NBA season.")
st.markdown("Use the sidebar to navigate to other analysis tools.")