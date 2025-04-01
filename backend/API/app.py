from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import sys
import os

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import our modules
from data_handler import (
    load_player_data, load_team_data, get_player_info, 
    get_player_game_logs, get_player_season_stats, get_player_headshot_url,
    get_team_logo_url
)
from prediction_models import PlayerStatPredictor
from clustering_models import PlayerClusterAnalyzer
from rl_models import RLCoachAgent

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize ML models
predictor = PlayerStatPredictor()
cluster_analyzer = PlayerClusterAnalyzer()
rl_agent = RLCoachAgent()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok"})

@app.route('/api/players', methods=['GET'])
def get_players():
    """Get all players for a season"""
    season = request.args.get('season', '2022-23')
    
    try:
        # Load player data
        players_df = load_player_data(season=season)
        
        # Convert to list of dictionaries
        players_list = []
        
        for _, player in players_df.iterrows():
            player_data = player.to_dict()
            # Add headshot URL
            player_data['HEADSHOT_URL'] = get_player_headshot_url(player_data['PLAYER_ID'])
            players_list.append(player_data)
            
        return jsonify({"players": players_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get all teams for a season"""
    season = request.args.get('season', '2022-23')
    
    try:
        # Load team data
        teams_df = load_team_data(season=season)
        
        # Convert to list of dictionaries
        teams_list = []
        
        for _, team in teams_df.iterrows():
            team_data = team.to_dict()
            # Add logo URL
            team_data['LOGO_URL'] = get_team_logo_url(team_data['TEAM_ID'])
            teams_list.append(team_data)
            
        return jsonify({"teams": teams_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/player/<int:player_id>/gamelog', methods=['GET'])
def get_player_gamelog(player_id):
    """Get game logs for a player"""
    season = request.args.get('season', '2022-23')
    
    try:
        # Get game logs
        game_logs_df = get_player_game_logs(player_id, season=season)
        
        # Convert to list of dictionaries
        game_logs = []
        
        for _, game in game_logs_df.iterrows():
            game_logs.append(game.to_dict())
            
        return jsonify({"game_logs": game_logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_player_stats():
    """Predict player statistics"""
    try:
        data = request.get_json()
        
        # Get required parameters
        player_id = data.get('player_id')
        features = data.get('features', {})
        model_type = data.get('model_type', 'random_forest')
        
        if not player_id:
            return jsonify({"error": "player_id is required"}), 400
            
        # Get player data to train the model
        season = '2022-23'
        players_df = load_player_data(season=season)
        
        # Train the model
        X, y = predictor.prepare_data(players_df)
        
        if model_type == 'linear_reg':
            predictor.train_linear_regression()
        else:
            predictor.train_random_forest()
            
        # Make prediction
        predictions = predictor.predict_stats(features, model_type=model_type)
        
        # Format response
        results = {}
        for stat, value in predictions.items():
            results[stat] = float(value)
            
        return jsonify({
            "player_id": player_id,
            "predictions": results,
            "model_type": model_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clustering', methods=['POST'])
def perform_clustering():
    """Perform player clustering"""
    try:
        data = request.get_json()
        
        # Get parameters
        n_clusters = data.get('n_clusters', 5)
        season = data.get('season', '2022-23')
        features = data.get('features', None)
        
        # Load player data
        players_df = load_player_data(season=season)
        
        # Perform clustering
        cluster_data = cluster_analyzer.prepare_data(players_df, features=features)
        cluster_analyzer.perform_clustering(n_clusters=n_clusters)
        
        # Get cluster centers and profiles
        centers = cluster_analyzer.get_cluster_centers()
        profiles = cluster_analyzer.get_cluster_profiles()
        players_by_cluster = cluster_analyzer.get_players_by_cluster(players_df)
        
        # Format response
        centers_dict = {}
        for i, center in enumerate(centers.to_dict('records')):
            centers_dict[f"cluster_{i}"] = center
            
        players_dict = {}
        for cluster, players_list in players_by_cluster.items():
            player_ids = [p['PLAYER_ID'] for p in players_list]
            players_dict[cluster] = player_ids
            
        return jsonify({
            "centers": centers_dict,
            "profiles": profiles,
            "players": players_dict,
            "n_clusters": n_clusters
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_game():
    """Simulate a game using RL models"""
    try:
        data = request.get_json()
        
        # Get parameters
        model_type = data.get('model_type', 'q_learning')
        initial_state = data.get('initial_state', None)
        max_steps = data.get('max_steps', 50)
        
        # Initialize RL agent if not done yet
        if model_type == 'q_learning' and not hasattr(rl_agent, 'q_model'):
            rl_agent.train_q_learning()
            model = rl_agent.q_model
        elif model_type == 'policy_gradient' and not hasattr(rl_agent, 'pg_model'):
            rl_agent.train_policy_gradient()
            model = rl_agent.pg_model
        else:
            model = rl_agent.q_model if model_type == 'q_learning' else rl_agent.pg_model
            
        # Simulate game
        history = rl_agent.simulate_game(model, initial_state=initial_state, max_steps=max_steps)
        
        # Format response
        results = {}
        for key, value in history.items():
            if hasattr(value, 'tolist'):
                results[key] = value.tolist()
            else:
                results[key] = value
                
        return jsonify({
            "simulation": results,
            "model_type": model_type
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For development - use Gunicorn in production
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)