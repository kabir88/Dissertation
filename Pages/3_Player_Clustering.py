import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_handler import load_player_data, get_player_headshot_url
from clustering_models import PlayerClusterAnalyzer

st.set_page_config(
    page_title="Player Clustering - NBA Strategy Optimization",
    page_icon="👥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .cluster-header {
        font-size: 1.8rem;
        color: #17408B;
        margin-bottom: 10px;
    }
    .cluster-subheader {
        font-size: 1.2rem;
        color: #E03A3E;
        margin-bottom: 20px;
    }
    .cluster-card {
        padding: 20px;
        border-radius: 5px;
        background-color: #f8f9fa;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        height: 100%;
    }
    .cluster-title {
        font-size: 1.3rem;
        font-weight: bold;
        color: #17408B;
        margin-bottom: 10px;
    }
    .player-item {
        font-size: 0.9rem;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Player Clustering")
st.sidebar.markdown("Group players into clusters based on playing style and statistics.")

# Load data
@st.cache_data(ttl=3600)
def load_data(season='2022-23'):
    return load_player_data(season)

# Initialize cluster analyzer
@st.cache_resource
def load_analyzer():
    return PlayerClusterAnalyzer()

players_df = load_data()
analyzer = load_analyzer()

# Main content
st.title("👥 Player Clustering Analysis")
st.markdown("Discover player archetypes and groupings based on statistical profiles using machine learning.")

# Season selector
available_seasons = ["2022-23", "2021-22", "2020-21", "2019-20", "2018-19"]
selected_season = st.selectbox("Select Season", available_seasons)

# Check if data is loaded
if players_df.empty:
    st.warning("Player data could not be loaded. Please check your connection to the NBA API.")
    st.stop()

# Filter players with significant minutes
min_minutes = st.slider("Minimum Minutes Per Game", 5, 30, 15)
filtered_players = players_df[players_df['MIN'] >= min_minutes].copy()

# Feature selection
st.subheader("Select Features for Clustering")

feature_options = [
    ('Scoring', ['PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT']),
    ('Rebounding', ['OREB', 'DREB', 'REB']),
    ('Playmaking', ['AST', 'TOV', 'AST_TOV']),
    ('Defense', ['STL', 'BLK', 'PF']),
    ('General', ['AGE', 'GP', 'MIN', 'PLUS_MINUS'])
]

selected_categories = st.multiselect(
    "Select statistical categories",
    [cat[0] for cat in feature_options],
    default=['Scoring', 'Rebounding', 'Playmaking']
)

# Extract features from selected categories
selected_features = []
for category in selected_categories:
    category_features = next((cat[1] for cat in feature_options if cat[0] == category), [])
    selected_features.extend(category_features)

# Remove any features not in the DataFrame
available_features = [feat for feat in selected_features if feat in filtered_players.columns]

if not available_features:
    st.warning("No valid features selected. Please choose different statistical categories.")
    st.stop()

# Display selected features
st.markdown(f"**Selected Features:** {', '.join(available_features)}")

# Tabs for different analysis
tab1, tab2, tab3 = st.tabs(["Cluster Analysis", "Player Explorer", "Similarity Search"])

with tab1:
    st.header("NBA Player Clustering")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Clustering parameters
        st.subheader("Clustering Parameters")
        
        # Option to automatically find optimal clusters
        auto_clusters = st.checkbox("Automatically find optimal number of clusters", value=True)
        
        if auto_clusters:
            max_clusters = st.slider("Maximum clusters to consider", 3, 15, 8)
        else:
            n_clusters = st.slider("Number of clusters", 2, 10, 5)
        
        # Apply PCA option
        apply_pca = st.checkbox("Apply PCA for dimensionality reduction", value=True)
        if apply_pca:
            n_components = st.slider("Number of principal components", 2, min(10, len(available_features)), 2)
        
        # Run clustering button
        if st.button("Run Clustering Analysis"):
            with st.spinner("Performing clustering analysis..."):
                # Prepare data
                scaled_data = analyzer.prepare_data(filtered_players, features=available_features)
                
                if scaled_data is None:
                    st.error("Error preparing data for clustering. Please check your dataset.")
                else:
                    # Apply PCA if selected
                    if apply_pca:
                        analyzer.apply_pca(n_components=n_components)
                    
                    # Find optimal number of clusters or use specified number
                    if auto_clusters:
                        optimal_n = analyzer.find_optimal_clusters(max_clusters=max_clusters)
                        st.success(f"Optimal number of clusters identified: {optimal_n}")
                        clusters = analyzer.perform_clustering(n_clusters=optimal_n)
                    else:
                        clusters = analyzer.perform_clustering(n_clusters=n_clusters)
                    
                    if clusters is not None:
                        st.success("Clustering completed successfully!")
                    else:
                        st.error("Clustering failed. Please try different parameters.")
    
    with col2:
        # Check if clustering has been performed
        if analyzer.clusters is not None:
            # Create 2D plot
            st.subheader("2D Cluster Visualization")
            fig = analyzer.plot_clusters_2d(df=filtered_players)
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Visualization is not available. Try applying PCA with 2 components.")
            
            # Feature comparison
            st.subheader("Feature Comparison")
            
            col1, col2 = st.columns(2)
            with col1:
                feature1 = st.selectbox("Select first feature", available_features, index=0)
            with col2:
                feature2 = st.selectbox("Select second feature", available_features, index=min(1, len(available_features)-1))
            
            if feature1 != feature2:
                fig = analyzer.plot_feature_comparison(feature1, feature2)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the clustering analysis using the panel on the left to see visualizations.")
    
    # Cluster profiles
    if analyzer.clusters is not None:
        st.header("Cluster Profiles")
        profiles = analyzer.get_cluster_profiles()
        
        if profiles:
            # Create a row for each cluster
            n_clusters = len(profiles)
            cols = st.columns(min(3, n_clusters))
            
            for i, (cluster_idx, profile) in enumerate(profiles.items()):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    st.markdown(
                        f"""
                        <div class="cluster-card">
                            <div class="cluster-title">Cluster {cluster_idx + 1}: {profile['description']}</div>
                            <p><strong>Players:</strong> {profile['size']}</p>
                            <p><strong>Avg. Points:</strong> {profile['avg_points']:.1f}</p>
                            <p><strong>Avg. Rebounds:</strong> {profile['avg_rebounds']:.1f}</p>
                            <p><strong>Avg. Assists:</strong> {profile['avg_assists']:.1f}</p>
                            <p><strong>Top Players:</strong></p>
                            <ul>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    for player in profile['top_players'][:5]:
                        st.markdown(f"<li class='player-item'>{player}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # Get cluster centers
            centers_df = analyzer.get_cluster_centers()
            
            if centers_df is not None:
                st.subheader("Cluster Centers")
                st.dataframe(centers_df.round(2))

with tab2:
    st.header("Player Explorer")
    st.markdown("Explore players by cluster to understand groupings and find similar players.")
    
    if analyzer.clusters is not None:
        # Get players by cluster
        cluster_groups = analyzer.get_players_by_cluster(filtered_players)
        
        if cluster_groups:
            # Cluster selection
            cluster_options = [f"Cluster {i+1}: {analyzer.get_cluster_profiles()[i]['description']}" 
                             for i in range(len(cluster_groups))]
            selected_cluster = st.selectbox("Select a cluster to explore", cluster_options)
            
            if selected_cluster:
                cluster_idx = int(selected_cluster.split(':')[0].replace('Cluster ', '')) - 1
                cluster_players = cluster_groups[cluster_idx]
                
                st.subheader(f"{selected_cluster}")
                st.markdown(f"**Number of players:** {len(cluster_players)}")
                
                # Display players in this cluster
                cols = st.columns(3)
                
                for i, (_, player) in enumerate(cluster_players.iterrows()):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="cluster-card">
                                <p><strong>{player['PLAYER_NAME']}</strong></p>
                                <p>{player['TEAM_ABBREVIATION']} | {player['POSITION']}</p>
                                <p>{player['PTS']:.1f} PPG, {player['REB']:.1f} RPG, {player['AST']:.1f} APG</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Display statistical averages for the cluster
                st.subheader("Cluster Statistics")
                
                cluster_stats = cluster_players[available_features].mean().reset_index()
                cluster_stats.columns = ['Statistic', 'Average']
                
                # Compare with league averages
                league_avgs = filtered_players[available_features].mean().reset_index()
                league_avgs.columns = ['Statistic', 'League Average']
                
                # Merge the DataFrames
                comparison = pd.merge(cluster_stats, league_avgs, on='Statistic')
                comparison['Difference'] = comparison['Average'] - comparison['League Average']
                comparison['Difference %'] = (comparison['Difference'] / comparison['League Average'] * 100).round(1)
                
                # Display the statistics
                st.dataframe(comparison.style.format({
                    'Average': '{:.2f}',
                    'League Average': '{:.2f}',
                    'Difference': '{:.2f}',
                    'Difference %': '{:.1f}%'
                }))
        else:
            st.info("No cluster groups available. Run the clustering analysis first.")
    else:
        st.info("Run the clustering analysis on the Cluster Analysis tab first.")

with tab3:
    st.header("Player Similarity Search")
    st.markdown("Find players with similar statistical profiles.")
    
    # Player selection
    all_players = sorted(filtered_players['PLAYER_NAME'].tolist())
    search_player = st.selectbox("Select a player", all_players, key="similarity_player")
    
    if search_player and analyzer.scaled_data is not None:
        # Get player index
        player_idx = filtered_players[filtered_players['PLAYER_NAME'] == search_player].index[0]
        
        # Get player data vector
        player_vector = analyzer.scaled_data[player_idx]
        
        # Calculate distances to all other players
        distances = []
        for i, player_name in enumerate(filtered_players['PLAYER_NAME']):
            if i != player_idx:
                # Calculate Euclidean distance
                dist = np.linalg.norm(analyzer.scaled_data[i] - player_vector)
                distances.append((player_name, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        # Display the player
        player_data = filtered_players[filtered_players['PLAYER_NAME'] == search_player].iloc[0]
        player_id = player_data['PLAYER_ID']
        
        st.subheader(f"Player Profile: {search_player}")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            player_img_url = get_player_headshot_url(player_id)
            st.image(player_img_url, width=160)
        
        with col2:
            st.markdown(f"**Team:** {player_data['TEAM_ABBREVIATION']} | **Position:** {player_data['POSITION']}")
            st.markdown(f"**Stats:** {player_data['PTS']:.1f} PPG, {player_data['REB']:.1f} RPG, {player_data['AST']:.1f} APG")
            
            if analyzer.clusters is not None:
                cluster_idx = player_data['cluster']
                cluster_desc = analyzer.get_cluster_profiles()[cluster_idx]['description']
                st.markdown(f"**Cluster:** {cluster_idx + 1} ({cluster_desc})")
        
        # Display the most similar players
        st.subheader("Most Similar Players")
        
        num_similar = st.slider("Number of similar players to display", 3, 15, 5)
        
        # Create columns for similar players
        cols = st.columns(min(5, num_similar))
        
        for i, (similar_player, distance) in enumerate(distances[:num_similar]):
            col_idx = i % len(cols)
            with cols[col_idx]:
                similar_data = filtered_players[filtered_players['PLAYER_NAME'] == similar_player].iloc[0]
                
                st.markdown(
                    f"""
                    <div class="cluster-card">
                        <p><strong>{similar_player}</strong></p>
                        <p>{similar_data['TEAM_ABBREVIATION']} | {similar_data['POSITION']}</p>
                        <p>{similar_data['PTS']:.1f} PPG, {similar_data['REB']:.1f} RPG, {similar_data['AST']:.1f} APG</p>
                        <p><em>Similarity: {1/(1+distance):.2f}</em></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        # Statistical comparison
        st.subheader("Statistical Comparison")
        
        # Select similar player for detailed comparison
        compare_player = st.selectbox("Select a player to compare with", [p[0] for p in distances[:10]])
        
        if compare_player:
            compare_data = filtered_players[filtered_players['PLAYER_NAME'] == compare_player].iloc[0]
            
            # Create comparison DataFrame
            comparison_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
            comparison_df = pd.DataFrame({
                'Statistic': comparison_stats,
                search_player: [player_data[stat] for stat in comparison_stats],
                compare_player: [compare_data[stat] for stat in comparison_stats]
            })
            
            # Calculate difference
            comparison_df['Difference'] = comparison_df[search_player] - comparison_df[compare_player]
            
            # Format percentages
            for i, stat in enumerate(comparison_df['Statistic']):
                if stat in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                    comparison_df.loc[i, search_player] = f"{comparison_df.loc[i, search_player]*100:.1f}%"
                    comparison_df.loc[i, compare_player] = f"{comparison_df.loc[i, compare_player]*100:.1f}%"
                    comparison_df.loc[i, 'Difference'] = f"{comparison_df.loc[i, 'Difference']*100:.1f}%"
            
            # Display comparison
            st.dataframe(comparison_df)
            
            # Radar chart for visual comparison
            categories = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks']
            
            # Normalize values for radar chart (0-1 scale)
            max_values = filtered_players[['PTS', 'REB', 'AST', 'STL', 'BLK']].max()
            
            player1_values = [player_data[stat] / max_values[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']]
            player2_values = [compare_data[stat] / max_values[stat] for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK']]
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=player1_values + [player1_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name=search_player
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
                title="Player Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the clustering analysis on the Cluster Analysis tab first.")

# Explanation of clustering
st.sidebar.markdown("---")
st.sidebar.subheader("About Clustering")
st.sidebar.markdown("""
**What is Clustering?**
Clustering is an unsupervised machine learning technique that groups similar data points together based on their features.

**How is it used here?**
We use K-means clustering to group NBA players based on their statistical profiles, identifying distinct player archetypes and playing styles.
""")

# Footer
st.markdown("---")
st.markdown("Data sourced from NBA Stats API. Clustering performed using scikit-learn.")
st.markdown("Use the sidebar to navigate to other analysis tools.")