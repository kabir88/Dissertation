import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from data_handler import preprocess_data_for_ml

class PlayerStatPredictor:
    """
    Class to predict player statistics using various ML models.
    """
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.features = None
        self.targets = None
        self.scaler = StandardScaler()
        self.linear_models = {}
        self.rf_models = {}
        
    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Prepare data for modeling by splitting into train and test sets.
        
        Args:
            df (pandas.DataFrame): DataFrame containing player statistics
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: X (features) and y (targets) for ML models
        """
        # Preprocess data
        X, y = preprocess_data_for_ml(df)
        
        if X is None or y is None:
            return None, None
            
        # Save feature and target names
        self.features = X.columns.tolist()
        self.targets = y.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self.X_train, self.y_train
        
    def train_linear_regression(self):
        """
        Train linear regression models for each target variable.
        
        Returns:
            dict: Dictionary of trained models
        """
        if self.X_train is None or self.y_train is None:
            return None
            
        self.linear_models = {}
        
        # Train model for each target
        for target in self.targets:
            model = LinearRegression()
            model.fit(self.X_train_scaled, self.y_train[target])
            self.linear_models[target] = model
            
        return self.linear_models
        
    def train_random_forest(self, n_estimators=100, max_depth=10):
        """
        Train random forest regression models for each target variable.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            
        Returns:
            dict: Dictionary of trained models
        """
        if self.X_train is None or self.y_train is None:
            return None
            
        self.rf_models = {}
        
        # Train model for each target
        for target in self.targets:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(self.X_train_scaled, self.y_train[target])
            self.rf_models[target] = model
            
        return self.rf_models
        
    def evaluate_models(self, model_type='linear_reg'):
        """
        Evaluate models of the specified type.
        
        Args:
            model_type (str): Type of model to evaluate ('linear_reg' or 'random_forest')
            
        Returns:
            pandas.DataFrame: DataFrame with evaluation metrics
        """
        if self.X_test is None or self.y_test is None:
            return None
            
        # Select appropriate models
        models = self.linear_models if model_type == 'linear_reg' else self.rf_models
        
        if not models:
            return None
            
        # Evaluate each model
        results = []
        for target, model in models.items():
            y_pred = model.predict(self.X_test_scaled)
            mse = mean_squared_error(self.y_test[target], y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test[target], y_pred)
            
            results.append({
                'Target': target,
                'RMSE': rmse,
                'R2': r2
            })
            
        return pd.DataFrame(results)
        
    def predict_stats(self, input_data, model_type='linear_reg'):
        """
        Predict player statistics using trained models.
        
        Args:
            input_data (dict or pandas.DataFrame): Player features
            model_type (str): Type of model to use ('linear_reg' or 'random_forest')
            
        Returns:
            dict: Predicted statistics
        """
        # Select appropriate models
        models = self.linear_models if model_type == 'linear_reg' else self.rf_models
        
        if not models:
            return None
            
        # Convert input to DataFrame if necessary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data.copy()
        
        # Ensure all required features are present
        for feature in self.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Scale input data
        input_scaled = self.scaler.transform(input_df[self.features])
        
        # Make predictions
        predictions = {}
        for target, model in models.items():
            pred = model.predict(input_scaled)
            predictions[target] = pred[0]
            
        return predictions
        
    def get_feature_importance(self, model_type='random_forest'):
        """
        Get feature importance from trained models.
        
        Args:
            model_type (str): Type of model to use (only 'random_forest' supported for importance)
            
        Returns:
            dict: Feature importance for each target
        """
        if model_type != 'random_forest' or not self.rf_models:
            return None
            
        importance_dict = {}
        for target, model in self.rf_models.items():
            # Get feature importance
            importances = model.feature_importances_
            
            # Create a mapping of features to importance
            importance_list = []
            for i, importance in enumerate(importances):
                importance_list.append({
                    'Feature': self.features[i],
                    'Importance': importance
                })
                
            # Sort by importance
            importance_df = pd.DataFrame(importance_list)
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            importance_dict[target] = importance_df
            
        return importance_dict
        
    def plot_predictions_vs_actual(self, target, model_type='linear_reg'):
        """
        Plot predicted vs actual values for a target variable.
        
        Args:
            target (str): Target variable to plot
            model_type (str): Type of model to use ('linear_reg' or 'random_forest')
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if self.X_test is None or self.y_test is None:
            return None
            
        # Select appropriate models
        models = self.linear_models if model_type == 'linear_reg' else self.rf_models
        
        if not models or target not in models:
            return None
            
        # Make predictions
        model = models[target]
        y_pred = model.predict(self.X_test_scaled)
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Actual': self.y_test[target],
            'Predicted': y_pred
        })
        
        # Create plot
        fig = px.scatter(
            plot_df,
            x='Actual',
            y='Predicted',
            title=f'Predicted vs Actual: {target}',
            labels={'Actual': f'Actual {target}', 'Predicted': f'Predicted {target}'}
        )
        
        # Add a perfect prediction line
        max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
        min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Perfect Prediction'
            )
        )
        
        return fig
        
    def plot_feature_importance(self, target, model_type='random_forest'):
        """
        Plot feature importance for a target variable.
        
        Args:
            target (str): Target variable to plot importance for
            model_type (str): Type of model to use (only 'random_forest' supported)
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if model_type != 'random_forest' or not self.rf_models:
            return None
            
        importance_dict = self.get_feature_importance(model_type='random_forest')
        
        if not importance_dict or target not in importance_dict:
            return None
            
        importance_df = importance_dict[target]
        
        # Get top 10 features
        top_n = min(10, len(importance_df))
        top_features = importance_df.head(top_n)
        
        # Create plot
        fig = px.bar(
            top_features,
            y='Feature',
            x='Importance',
            orientation='h',
            title=f'Feature Importance for {target}',
            labels={'Importance': 'Importance', 'Feature': 'Feature'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400
        )
        
        return fig