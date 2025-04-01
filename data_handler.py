import pandas as pd
import numpy as np
import os
import json
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats, playergamelog, playercareerstats
from nba_api.stats.static import players, teams
import time
from backend.utils.db import cache_nba_data, get_cached_nba_data

def load_player_data(season='2022-23'):
    """
    Load player statistics for the given season using the NBA API.
    
    Args:
        season (str): The NBA season to fetch data for
        
    Returns:
        pandas.DataFrame: DataFrame containing player statistics
    """
    try:
        # Try to get cached data first
        cached_data = get_cached_nba_data('players', season)
        if cached_data is not None and len(cached_data) > 0:
            return pd.DataFrame(cached_data)
            
        print("Fetching player data from NBA API...")
        # If no cached data, fetch from API
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
            plus_minus='N',
            pace_adjust='N',
            rank='N',
            month=0,
            last_n_games=0
        )
        
        # Convert to DataFrame
        df = player_stats.get_data_frames()[0]
        
        # Filter for players with significant minutes
        df = df[df['MIN'] > 10]
        
        # Cache the data
        cache_nba_data('players', season, df.to_dict('records'))
        
        return df
    except Exception as e:
        print(f"Error loading player data: {e}")
        # Return an empty DataFrame if there's an error
        return pd.DataFrame()

def load_team_data(season='2022-23'):
    """
    Load team statistics for the given season using the NBA API.
    
    Args:
        season (str): The NBA season to fetch data for
        
    Returns:
        pandas.DataFrame: DataFrame containing team statistics
    """
    try:
        # Try to get cached data first
        cached_data = get_cached_nba_data('teams', season)
        if cached_data is not None and len(cached_data) > 0:
            return pd.DataFrame(cached_data)
            
        print("Fetching team data from NBA API...")
        # If no cached data, fetch from API
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='PerGame',
            measure_type_detailed_defense='Base',
            plus_minus='N',
            pace_adjust='N',
            rank='N',
            last_n_games=0
        )
        
        # Convert to DataFrame
        df = team_stats.get_data_frames()[0]
        
        # Cache the data
        cache_nba_data('teams', season, df.to_dict('records'))
        
        return df
    except Exception as e:
        print(f"Error loading team data: {e}")
        # Return an empty DataFrame if there's an error
        return pd.DataFrame()

def get_player_headshot_url(player_id):
    """
    Get the URL for a player's headshot image.
    
    Args:
        player_id (int): The NBA API player ID
        
    Returns:
        str: URL to the player's headshot image
    """
    return f"https://ak-static.cms.nba.com/wp-content/uploads/headshots/nba/latest/260x190/{player_id}.png"

def get_team_logo_url(team_id):
    """
    Get the URL for a team's logo.
    
    Args:
        team_id (int): The NBA API team ID
        
    Returns:
        str: URL to the team's logo
    """
    return f"https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.svg"

def get_player_info(player_name):
    """
    Get detailed information for a player by name.
    
    Args:
        player_name (str): The player's name
        
    Returns:
        dict: Player information including ID, team, etc.
    """
    try:
        # Search for the player
        player_list = players.find_players_by_full_name(player_name)
        
        if not player_list:
            return None
            
        # Get the first match
        player = player_list[0]
        return player
    except Exception as e:
        print(f"Error getting player info: {e}")
        return None

def get_player_game_logs(player_id, season='2022-23'):
    """
    Get game logs for a player for the given season.
    
    Args:
        player_id (int): The NBA API player ID
        season (str): The NBA season to fetch data for
        
    Returns:
        pandas.DataFrame: DataFrame containing game logs
    """
    try:
        # Try to get cached data first
        cache_key = f"player_gamelogs_{player_id}"
        cached_data = get_cached_nba_data(cache_key, season)
        if cached_data is not None and len(cached_data) > 0:
            return pd.DataFrame(cached_data)
            
        # If no cached data, fetch from API
        game_logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star='Regular Season'
        )
        
        # Convert to DataFrame
        df = game_logs.get_data_frames()[0]
        
        # Add a running average column for key stats
        for stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']:
            if stat in df.columns:
                df[f'{stat}_AVG'] = df[stat].expanding().mean()
        
        # Convert game date to datetime
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Cache the data
        cache_nba_data(cache_key, season, df.to_dict('records'))
        
        # Rate limiting to avoid API rate limits
        time.sleep(0.6)
        
        return df
    except Exception as e:
        print(f"Error getting player game logs: {e}")
        # Return an empty DataFrame if there's an error
        return pd.DataFrame()

def get_player_season_stats(player_id, seasons=None):
    """
    Get season statistics for a player across multiple seasons.
    
    Args:
        player_id (int): The NBA API player ID
        seasons (list): List of seasons to fetch data for, defaults to last 3 seasons
        
    Returns:
        pandas.DataFrame: DataFrame containing season stats
    """
    try:
        # If seasons not provided, use the last 3 seasons
        if seasons is None:
            current_year = 2022  # Update this to the current year when needed
            seasons = [f"{year-1}-{str(year)[-2:]}" for year in range(current_year-2, current_year+1)]
        
        # Try to get cached data first
        cache_key = f"player_seasons_{player_id}"
        cache_season = "_".join(seasons)
        cached_data = get_cached_nba_data(cache_key, cache_season)
        if cached_data is not None and len(cached_data) > 0:
            return pd.DataFrame(cached_data)
            
        # If no cached data, fetch from API
        career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
        
        # Convert to DataFrame
        df = career_stats.get_data_frames()[0]
        
        # Filter for regular season and requested seasons
        df = df[df['SEASON_ID'].isin([f"2{season.replace('-', '')}" for season in seasons])]
        
        # Cache the data
        cache_nba_data(cache_key, cache_season, df.to_dict('records'))
        
        # Rate limiting to avoid API rate limits
        time.sleep(0.6)
        
        return df
    except Exception as e:
        print(f"Error getting player season stats: {e}")
        # Return an empty DataFrame if there's an error
        return pd.DataFrame()

def preprocess_data_for_ml(df):
    """
    Preprocess player data for machine learning models.
    
    Args:
        df (pandas.DataFrame): DataFrame containing player statistics
        
    Returns:
        tuple: X (features) and y (targets) for ML models
    """
    if df.empty:
        return None, None
        
    # Select features
    features = [
        'AGE', 'GP', 'W', 'L', 'MIN', 'FGM', 'FGA', 'FG_PCT',
        'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
        'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF'
    ]
    
    # Select targets
    targets = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    
    # Create feature and target dataframes
    X = df[features].copy()
    y = df[targets].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)
    
    return X, y