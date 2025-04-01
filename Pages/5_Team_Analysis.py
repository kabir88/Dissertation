import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_team_data, load_player_data, get_team_logo_url

st.title("Team Analysis")
st.write("Analyze team performance, composition, and strategic patterns.")

# Sidebar for team selection
st.sidebar.header("Team Selection")

# Season selection
season = st.sidebar.selectbox(
    "Select Season",
    ["2022-23", "2021-22", "2020-21"]
)

# Load data
with st.spinner("Loading data..."):
    try:
        # Load team data
        teams_df = load_team_data(season)
        
        if teams_df.empty:
            st.error("No team data available. Please check your internet connection or try again later.")
            st.stop()
            
        # Load player data for roster analysis
        players_df = load_player_data(season)
        
        # Team selection
        selected_team = st.sidebar.selectbox(
            "Select Team",
            sorted(teams_df["TEAM_NAME"].tolist())
        )
        
        # Get team details
        team_data = teams_df[teams_df["TEAM_NAME"] == selected_team].iloc[0]
        team_id = team_data["TEAM_ID"]
        
        # Team logo and basic info
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Team logo
            logo_url = get_team_logo_url(team_id)
            st.image(logo_url, width=150)
            
        with col2:
            # Key team stats
            st.subheader(f"{selected_team} ({season})")
            
            # Season summary
            wins = team_data["W"]
            losses = team_data["L"]
            win_pct = team_data["W_PCT"]
            
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Record", f"{wins}-{losses}")
            metrics_cols[1].metric("Win %", f"{win_pct:.3f}")
            
            # Determine playoff status (simplified)
            if win_pct >= 0.5:
                playoff_status = "Playoff Team"
            else:
                playoff_status = "Lottery Team"
                
            metrics_cols[2].metric("Status", playoff_status)
        
        # Team performance dashboard
        st.header("Team Performance")
        
        # Key stats tabs
        tab1, tab2, tab3 = st.tabs(["Offensive Stats", "Defensive Stats", "Advanced Metrics"])
        
        with tab1:
            # Offensive stats
            off_cols = st.columns(3)
            off_cols[0].metric("Points Per Game", f"{team_data['PTS']:.1f}")
            off_cols[1].metric("Field Goal %", f"{team_data['FG_PCT']:.3f}")
            off_cols[2].metric("3-Point %", f"{team_data['FG3_PCT']:.3f}")
            
            off_cols2 = st.columns(3)
            off_cols2[0].metric("Assists Per Game", f"{team_data['AST']:.1f}")
            off_cols2[1].metric("Offensive Rebounds", f"{team_data['OREB']:.1f}")
            off_cols2[2].metric("Turnovers", f"{team_data['TOV']:.1f}")
            
            # Offensive radar chart
            off_features = ['PTS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'AST', 'OREB', 'TOV']
            
            # Get league averages
            league_avg = teams_df[off_features].mean()
            
            # Normalize to percentiles
            team_percentiles = []
            for feature in off_features:
                if feature in ['TOV']:  # Lower is better
                    percentile = (teams_df[feature] >= team_data[feature]).mean() * 100
                else:  # Higher is better
                    percentile = (teams_df[feature] <= team_data[feature]).mean() * 100
                team_percentiles.append(percentile / 100)
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=team_percentiles,
                theta=off_features,
                fill='toself',
                name=selected_team
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Offensive Profile (Percentile Rank)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            # Defensive stats
            def_cols = st.columns(3)
            def_cols[0].metric("Opponent Points", f"{team_data['OPP_PTS']:.1f}")
            def_cols[1].metric("Blocks Per Game", f"{team_data['BLK']:.1f}")
            def_cols[2].metric("Steals Per Game", f"{team_data['STL']:.1f}")
            
            def_cols2 = st.columns(3)
            def_cols2[0].metric("Defensive Rebounds", f"{team_data['DREB']:.1f}")
            def_cols2[1].metric("Personal Fouls", f"{team_data['PF']:.1f}")
            
            # Point differential
            point_diff = team_data['PTS'] - team_data['OPP_PTS']
            def_cols2[2].metric("Point Differential", f"{point_diff:.1f}", 
                              f"{point_diff:.1f}" if point_diff >= 0 else f"{point_diff:.1f}")
            
            # Defensive radar chart
            def_features = ['OPP_PTS', 'DREB', 'STL', 'BLK', 'PF']
            
            # Get league averages
            league_avg = teams_df[def_features].mean()
            
            # Normalize to percentiles
            team_percentiles = []
            for feature in def_features:
                if feature in ['OPP_PTS', 'PF']:  # Lower is better
                    percentile = (teams_df[feature] >= team_data[feature]).mean() * 100
                else:  # Higher is better
                    percentile = (teams_df[feature] <= team_data[feature]).mean() * 100
                team_percentiles.append(percentile / 100)
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=team_percentiles,
                theta=def_features,
                fill='toself',
                name=selected_team
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Defensive Profile (Percentile Rank)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            # Advanced metrics
            adv_cols = st.columns(3)
            adv_cols[0].metric("Pace", f"{team_data['PACE']:.1f}")
            adv_cols[1].metric("Offensive Rating", f"{team_data['OFF_RATING']:.1f}")
            adv_cols[2].metric("Defensive Rating", f"{team_data['DEF_RATING']:.1f}")
            
            adv_cols2 = st.columns(3)
            adv_cols2[0].metric("Net Rating", f"{team_data['NET_RATING']:.1f}")
            adv_cols2[1].metric("Assist Ratio", f"{team_data['AST_RATIO']:.1f}")
            adv_cols2[2].metric("Turnover Ratio", f"{team_data['TM_TOV_PCT']:.3f}")
            
            # Create a Four Factors chart (simplified)
            four_factors = pd.DataFrame({
                'Factor': ['Shooting (eFG%)', 'Turnovers (TOV%)', 'Rebounding (OREB%)', 'Free Throws (FT Rate)'],
                'Value': [team_data['EFG_PCT'], team_data['TM_TOV_PCT'] * -1, team_data['OREB_PCT'], team_data['FTA_RATE']],
                'League Rank': [
                    (teams_df['EFG_PCT'] <= team_data['EFG_PCT']).mean() * 30,
                    (teams_df['TM_TOV_PCT'] >= team_data['TM_TOV_PCT']).mean() * 30,
                    (teams_df['OREB_PCT'] <= team_data['OREB_PCT']).mean() * 30,
                    (teams_df['FTA_RATE'] <= team_data['FTA_RATE']).mean() * 30
                ]
            })
            
            # Create horizontal bar chart
            fig = px.bar(
                four_factors,
                y='Factor',
                x='League Rank',
                orientation='h',
                title="Four Factors (League Rank 1-30)",
                labels={'League Rank': 'League Rank (1 = Best)', 'Factor': 'Factor'},
                range_x=[30, 0],  # Reverse scale so 1 is best
                color='League Rank',
                color_continuous_scale='RdYlGn_r'  # Reversed so green is better
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Team roster analysis
        st.header("Team Roster Analysis")
        
        # Filter players on the selected team
        team_abbrev = team_data["TEAM_ABBREVIATION"]
        team_players = players_df[players_df["TEAM_ABBREVIATION"] == team_abbrev].copy()
        
        if team_players.empty:
            st.warning(f"No player data available for {selected_team}.")
        else:
            # Sort by minutes played
            team_players = team_players.sort_values(by="MIN", ascending=False)
            
            # Show roster table
            st.subheader("Roster")
            roster_cols = ['PLAYER_NAME', 'AGE', 'POSITION', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLUS_MINUS']
            st.dataframe(team_players[roster_cols])
            
            # Scoring distribution
            st.subheader("Scoring Distribution")
            
            # Create a pie chart of point distribution
            team_players['TOTAL_PTS'] = team_players['PTS'] * team_players['GP']
            
            # Get top 8 players by total points
            top_scorers = team_players.nlargest(8, 'TOTAL_PTS')
            
            # Add 'Others' category
            others_pts = team_players.loc[~team_players['PLAYER_NAME'].isin(top_scorers['PLAYER_NAME']), 'TOTAL_PTS'].sum()
            
            # Create the data for the pie chart
            if others_pts > 0:
                pts_data = pd.concat([
                    top_scorers[['PLAYER_NAME', 'TOTAL_PTS']],
                    pd.DataFrame({'PLAYER_NAME': ['Others'], 'TOTAL_PTS': [others_pts]})
                ])
            else:
                pts_data = top_scorers[['PLAYER_NAME', 'TOTAL_PTS']]
            
            # Create pie chart
            fig = px.pie(
                pts_data,
                names='PLAYER_NAME',
                values='TOTAL_PTS',
                title="Team Scoring Distribution",
                hole=0.4
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Minutes distribution
            st.subheader("Playing Time Analysis")
            
            # Calculate minutes per game for players with significant time
            sig_players = team_players[team_players['MIN'] >= 10].copy()
            sig_players['MIN_TOTAL'] = sig_players['MIN'] * sig_players['GP']
            
            # Sort by total minutes
            sig_players = sig_players.sort_values('MIN_TOTAL', ascending=False)
            
            # Create horizontal bar chart
            fig = px.bar(
                sig_players.head(10),
                y='PLAYER_NAME',
                x='MIN',
                orientation='h',
                title="Minutes Per Game (Top 10 Players)",
                labels={'MIN': 'Minutes Per Game', 'PLAYER_NAME': 'Player'},
                color='MIN',
                color_continuous_scale='Blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Player efficiency
            st.subheader("Player Efficiency")
            
            # Create efficiency scatter plot (PTS vs. MIN)
            fig = px.scatter(
                team_players[team_players['MIN'] >= 5],
                x='MIN',
                y='PTS',
                size='GP',
                hover_name='PLAYER_NAME',
                text='PLAYER_NAME',
                title="Scoring Efficiency: Points vs. Minutes Played",
                labels={'MIN': 'Minutes Per Game', 'PTS': 'Points Per Game', 'GP': 'Games Played'}
            )
            
            # Add trend line
            fig.add_trace(
                go.Scatter(
                    x=[10, max(team_players['MIN'])],
                    y=[10, max(team_players['MIN'])],
                    mode='lines',
                    line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
                    name='1 Point Per Minute'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Shooting distribution
            st.subheader("Shooting Distribution")
            
            # Filter players with significant minutes
            shooters = team_players[team_players['MIN'] >= 15].copy()
            
            if not shooters.empty:
                # Calculate field goal attempts distribution
                shooters['FGA_PCT'] = shooters['FGA'] / shooters['FGA'].sum()
                shooters['FG3A_PCT'] = shooters['FG3A'] / shooters['FG3A'].sum()
                shooters['FTA_PCT'] = shooters['FTA'] / shooters['FTA'].sum()
                
                # Calculate total shots
                shooters['TOTAL_SHOTS'] = shooters['FGA'] * shooters['GP']
                
                # Sort by total shots
                shooters = shooters.sort_values('TOTAL_SHOTS', ascending=False).head(8)
                
                # Create field goal attempts distribution
                fig = px.bar(
                    shooters,
                    x='PLAYER_NAME',
                    y=['FG2A', 'FG3A'],
                    title="Shot Distribution by Player",
                    labels={'value': 'Attempts Per Game', 'PLAYER_NAME': 'Player', 'variable': 'Shot Type'},
                    barmode='stack'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create a scatter plot of FG% vs. 3P%
                fig = px.scatter(
                    shooters,
                    x='FG_PCT',
                    y='FG3_PCT',
                    size='FGA',
                    hover_name='PLAYER_NAME',
                    text='PLAYER_NAME',
                    title="Shooting Percentages: FG% vs. 3P%",
                    labels={'FG_PCT': 'Field Goal %', 'FG3_PCT': '3-Point %', 'FGA': 'Field Goal Attempts Per Game'}
                )
                
                # Add quadrant divisions
                fg_avg = shooters['FG_PCT'].mean()
                fg3_avg = shooters['FG3_PCT'].mean()
                
                fig.add_vline(x=fg_avg, line_dash="dash", line_color="gray")
                fig.add_hline(y=fg3_avg, line_dash="dash", line_color="gray")
                
                # Add quadrant labels
                fig.add_annotation(x=fg_avg/2, y=fg3_avg/2, text="Below Average", showarrow=False)
                fig.add_annotation(x=fg_avg/2, y=fg3_avg*1.5, text="3PT Specialist", showarrow=False)
                fig.add_annotation(x=fg_avg*1.5, y=fg3_avg/2, text="Inside Scorer", showarrow=False)
                fig.add_annotation(x=fg_avg*1.5, y=fg3_avg*1.5, text="Elite Shooter", showarrow=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
        # League-wide comparison
        st.header("League-Wide Comparison")
        
        # Team rankings table
        st.subheader("Team Rankings")
        
        # Calculate rankings for key stats
        key_stats = ['PTS', 'OPP_PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'NET_RATING']
        rankings = {}
        
        for stat in key_stats:
            if stat in ['OPP_PTS', 'TOV']:  # Lower is better
                rankings[stat] = (teams_df[stat] < team_data[stat]).sum() + 1
            else:  # Higher is better
                rankings[stat] = (teams_df[stat] > team_data[stat]).sum() + 1
                
        # Create rankings dataframe
        rankings_df = pd.DataFrame({
            'Statistic': [
                'Points Per Game', 'Opponent Points', 'Rebounds', 'Assists', 
                'Steals', 'Blocks', 'Turnovers', 'FG%', '3PT%', 'FT%', 'Net Rating'
            ],
            'Value': [team_data[stat] for stat in key_stats],
            'League Rank': [rankings[stat] for stat in key_stats]
        })
        
        # Create a horizontal bar chart for rankings
        fig = px.bar(
            rankings_df,
            y='Statistic',
            x='League Rank',
            orientation='h',
            title="League Rankings (1 = Best)",
            labels={'League Rank': 'League Rank', 'Statistic': 'Statistic'},
            range_x=[30, 0],  # Reverse scale so 1 is best
            color='League Rank',
            color_continuous_scale='RdYlGn_r'  # Reversed so green is better
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Comparison with specific teams
        st.subheader("Team Comparison")
        
        # Select teams to compare with
        compare_teams = st.multiselect(
            "Select teams to compare with",
            [t for t in sorted(teams_df["TEAM_NAME"].tolist()) if t != selected_team],
            default=[t for t in sorted(teams_df.nlargest(3, 'W_PCT')["TEAM_NAME"].tolist()) if t != selected_team][:2]
        )
        
        if compare_teams:
            # Add the selected team
            all_compare_teams = [selected_team] + compare_teams
            
            # Filter the dataframe
            compare_df = teams_df[teams_df["TEAM_NAME"].isin(all_compare_teams)]
            
            # Select stats to compare
            compare_stats = st.multiselect(
                "Select statistics to compare",
                ['PTS', 'OPP_PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'NET_RATING'],
                default=['PTS', 'OPP_PTS', 'NET_RATING']
            )
            
            if compare_stats:
                # Create comparison dataframe
                plot_df = pd.melt(
                    compare_df, 
                    id_vars=["TEAM_NAME"], 
                    value_vars=compare_stats,
                    var_name="Statistic", 
                    value_name="Value"
                )
                
                # Create grouped bar chart
                fig = px.bar(
                    plot_df,
                    x="Statistic",
                    y="Value",
                    color="TEAM_NAME",
                    barmode="group",
                    title="Team Comparison",
                    labels={"Value": "Value", "Statistic": "Statistic", "TEAM_NAME": "Team"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("There was an error processing the data. Please try again later.")