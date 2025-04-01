import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from data_handler import preprocess_data_for_ml

class PlayerClusterAnalyzer:
    """
    Class to cluster players based on their performance metrics.
    """
    def __init__(self):
        self.data = None
        self.scaled_data = None
        self.pca_data = None
        self.clusters = None
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.features = None
        self.player_data = None
        
    def prepare_data(self, df, features=None):
        """
        Prepare data for clustering.
        
        Args:
            df (pandas.DataFrame): DataFrame containing player statistics
            features (list): List of features to use for clustering
            
        Returns:
            numpy.ndarray: Scaled features for clustering
        """
        if df.empty:
            return None
            
        # Save the original data
        self.player_data = df.copy()
        
        # Default features if none provided
        if features is None:
            features = [
                'AGE', 'GP', 'MIN', 'FGM', 'FGA', 'FG_PCT',
                'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS'
            ]
            
        # Filter for available features
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            return None
            
        # Save feature names
        self.features = available_features
        
        # Extract features
        self.data = df[available_features].copy()
        
        # Handle missing values
        self.data.fillna(self.data.mean(), inplace=True)
        
        # Scale data
        self.scaled_data = self.scaler.fit_transform(self.data)
        
        return self.scaled_data
        
    def apply_pca(self, n_components=2):
        """
        Apply PCA to reduce dimensionality.
        
        Args:
            n_components (int): Number of principal components
            
        Returns:
            numpy.ndarray: PCA-transformed data
        """
        if self.scaled_data is None:
            return None
            
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
        
        return self.pca_data
        
    def find_optimal_clusters(self, max_clusters=10):
        """
        Find the optimal number of clusters using silhouette score.
        
        Args:
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            int: Optimal number of clusters
        """
        if self.scaled_data is None:
            return None
            
        # Data to use for clustering
        data = self.pca_data if self.pca_data is not None else self.scaled_data
        
        # Calculate silhouette score for different numbers of clusters
        silhouette_scores = []
        
        # Test from 2 to max_clusters
        for n_clusters in range(2, max_clusters + 1):
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
        # Find the optimal number of clusters
        if not silhouette_scores:
            return 5  # Default to 5 clusters if silhouette analysis fails
            
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        
        return optimal_clusters
        
    def perform_clustering(self, n_clusters=5):
        """
        Perform KMeans clustering.
        
        Args:
            n_clusters (int): Number of clusters
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.scaled_data is None:
            return None
            
        # Data to use for clustering
        data = self.pca_data if self.pca_data is not None else self.scaled_data
        
        # Apply KMeans
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = self.kmeans.fit_predict(data)
        
        # Add cluster labels to player data if available
        if self.player_data is not None:
            self.player_data['cluster'] = self.clusters
            
        return self.clusters
        
    def get_cluster_centers(self):
        """
        Get the cluster centers in the original feature space.
        
        Returns:
            pandas.DataFrame: Cluster centers with feature names
        """
        if self.kmeans is None or self.features is None:
            return None
            
        # Get the cluster centers
        centers = self.kmeans.cluster_centers_
        
        # If PCA was applied, transform back to original space
        if self.pca is not None:
            # This is an approximation since PCA is not perfectly invertible
            centers = self.pca.inverse_transform(centers)
            
        # Inverse transform to get original scale
        centers = self.scaler.inverse_transform(centers)
        
        # Create DataFrame with feature names
        centers_df = pd.DataFrame(centers, columns=self.features)
        centers_df['cluster'] = range(len(centers))
        
        return centers_df
        
    def get_cluster_profiles(self):
        """
        Create descriptive profiles for each cluster.
        
        Returns:
            dict: Cluster profiles with descriptions
        """
        if self.kmeans is None or self.player_data is None:
            return None
            
        # Get cluster centers
        centers_df = self.get_cluster_centers()
        
        if centers_df is None:
            return None
            
        # Create profiles
        profiles = {}
        
        for cluster_idx in range(len(self.kmeans.cluster_centers_)):
            # Get players in this cluster
            cluster_players = self.player_data[self.player_data['cluster'] == cluster_idx]
            
            # Get center for this cluster
            center = centers_df[centers_df['cluster'] == cluster_idx].iloc[0]
            
            # Create profile description
            profile = {
                'size': len(cluster_players),
                'avg_points': center['PTS'] if 'PTS' in center else None,
                'avg_rebounds': center['REB'] if 'REB' in center else None,
                'avg_assists': center['AST'] if 'AST' in center else None,
                'avg_age': center['AGE'] if 'AGE' in center else None,
                'avg_minutes': center['MIN'] if 'MIN' in center else None,
                'top_players': cluster_players.nlargest(5, 'MIN')['PLAYER_NAME'].tolist() if 'PLAYER_NAME' in cluster_players.columns and 'MIN' in cluster_players.columns else [],
                'description': self._generate_cluster_description(center)
            }
            
            profiles[cluster_idx] = profile
            
        return profiles
        
    def _generate_cluster_description(self, center):
        """
        Generate a descriptive label for a cluster based on center values.
        
        Args:
            center (pandas.Series): Cluster center data
            
        Returns:
            str: Descriptive label
        """
        descriptions = []
        
        # Check for scoring
        if 'PTS' in center:
            if center['PTS'] > 20:
                descriptions.append("High Scorer")
            elif center['PTS'] < 10:
                descriptions.append("Low Scorer")
                
        # Check for playmaking
        if 'AST' in center:
            if center['AST'] > 7:
                descriptions.append("Playmaker")
            elif center['AST'] > 4:
                descriptions.append("Secondary Playmaker")
                
        # Check for rebounding
        if 'REB' in center:
            if center['REB'] > 10:
                descriptions.append("Elite Rebounder")
            elif center['REB'] > 7:
                descriptions.append("Strong Rebounder")
                
        # Check for defense
        if 'STL' in center and 'BLK' in center:
            if center['STL'] + center['BLK'] > 2.5:
                descriptions.append("Defensive Specialist")
                
        # Check for shooting
        if 'FG3_PCT' in center and 'FG3A' in center:
            if center['FG3_PCT'] > 0.38 and center['FG3A'] > 4:
                descriptions.append("Sharpshooter")
                
        # Check for efficiency
        if 'FG_PCT' in center:
            if center['FG_PCT'] > 0.55 and ('PTS' in center and center['PTS'] > 10):
                descriptions.append("Efficient Scorer")
                
        # Check for role players
        if not descriptions and 'MIN' in center:
            if center['MIN'] < 20:
                descriptions.append("Role Player")
            else:
                descriptions.append("Balanced Contributor")
                
        # Default description
        if not descriptions:
            descriptions.append("Balanced Player")
            
        return ", ".join(descriptions)
        
    def get_players_by_cluster(self, df):
        """
        Get players grouped by cluster.
        
        Args:
            df (pandas.DataFrame): Original player data
            
        Returns:
            dict: Players grouped by cluster
        """
        if self.clusters is None:
            return None
            
        # Create a copy of the data with cluster labels
        clustered_df = df.copy()
        clustered_df['cluster'] = self.clusters
        
        # Group by cluster
        cluster_groups = {cluster: clustered_df[clustered_df['cluster'] == cluster] 
                         for cluster in range(len(self.kmeans.cluster_centers_))}
        
        return cluster_groups
        
    def plot_clusters_2d(self, df=None):
        """
        Plot clusters in 2D (using PCA if needed).
        
        Args:
            df (pandas.DataFrame): Original player data
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if self.clusters is None:
            return None
            
        # Data to use for plotting
        if self.pca_data is None:
            # Apply PCA to get 2D representation
            self.apply_pca(n_components=2)
            
        # Prepare data for plotting
        plot_data = pd.DataFrame({
            'PC1': self.pca_data[:, 0],
            'PC2': self.pca_data[:, 1],
            'Cluster': self.clusters
        })
        
        # Add player names if available
        if df is not None and 'PLAYER_NAME' in df.columns:
            plot_data['Player'] = df['PLAYER_NAME'].values
            
        # Create plot
        fig = px.scatter(
            plot_data,
            x='PC1',
            y='PC2',
            color='Cluster',
            hover_name='Player' if 'Player' in plot_data.columns else None,
            title='Player Clusters',
            color_continuous_scale=px.colors.qualitative.G10,
            labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
        )
        
        # Add variance explained by each component
        if self.pca is not None:
            fig.update_layout(
                title=f'Player Clusters (PC1: {self.pca.explained_variance_ratio_[0]:.2%} variance, PC2: {self.pca.explained_variance_ratio_[1]:.2%} variance)'
            )
            
        return fig
        
    def plot_feature_comparison(self, feature1, feature2):
        """
        Plot two features with cluster coloring.
        
        Args:
            feature1 (str): First feature to plot
            feature2 (str): Second feature to plot
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if self.clusters is None or self.player_data is None:
            return None
            
        # Check if features are available
        if feature1 not in self.player_data.columns or feature2 not in self.player_data.columns:
            return None
            
        # Prepare data for plotting
        plot_data = pd.DataFrame({
            feature1: self.player_data[feature1],
            feature2: self.player_data[feature2],
            'Cluster': self.clusters
        })
        
        # Add player names if available
        if 'PLAYER_NAME' in self.player_data.columns:
            plot_data['Player'] = self.player_data['PLAYER_NAME']
            
        # Create plot
        fig = px.scatter(
            plot_data,
            x=feature1,
            y=feature2,
            color='Cluster',
            hover_name='Player' if 'Player' in plot_data.columns else None,
            title=f'{feature1} vs {feature2} by Cluster',
            color_continuous_scale=px.colors.qualitative.G10
        )
        
        return fig