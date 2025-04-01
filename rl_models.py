import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import random
import plotly.express as px
import plotly.graph_objects as go

# Note: This is a simplified version of RL implementation
# The full version would use stable_baselines3 library
# from stable_baselines3 import DQN, PPO
# from stable_baselines3.common.evaluation import evaluate_policy

class BasketballDecisionEnv(gym.Env):
    """
    A simplified basketball decision-making environment for reinforcement learning.
    
    State space:
    - score_diff: Point difference (team - opponent)
    - time_remaining: Time remaining in the game (minutes)
    - shot_clock: Shot clock (seconds)
    - court_position: Position on court (0-1, where 0 is own basket, 1 is opponent basket)
    - player_performance: Performance rating of available players (normalized 0-1)
    
    Action space:
    - 0: Pass
    - 1: Drive to basket
    - 2: Shoot 2-pointer
    - 3: Shoot 3-pointer
    - 4: Call for screen
    - 5: Call timeout
    """
    def __init__(self):
        super(BasketballDecisionEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(6)
        
        # 5 continuous state variables
        self.observation_space = spaces.Box(
            low=np.array([-50, 0, 0, 0, 0]),
            high=np.array([50, 48, 24, 1, 1]),
            dtype=np.float32
        )
        
        # Game state
        self.state = None
        self.done = False
        self.steps = 0
        self.max_steps = 100
        self.history = {
            'states': [],
            'actions': [],
            'rewards': [],
            'score_diff': []
        }
        
        # Action success probabilities based on player performance and situation
        self.base_success_prob = {
            0: 0.95,  # Pass
            1: 0.6,   # Drive
            2: 0.5,   # 2-pointer
            3: 0.35,  # 3-pointer
            4: 0.85,  # Screen
            5: 1.0    # Timeout
        }
        
    def reset(self, seed=None):
        """
        Reset the environment to initial state.
        
        Returns:
            numpy.ndarray: Initial observation
            dict: Additional information
        """
        super().reset(seed=seed)
        
        # Initial state: score_diff, time_remaining, shot_clock, court_position, player_performance
        self.state = np.array([
            0,                     # Even score
            48.0,                  # Full game (48 minutes)
            24.0,                  # Full shot clock
            random.uniform(0, 1),  # Random court position
            random.uniform(0.4, 0.9)  # Random player performance
        ], dtype=np.float32)
        
        self.done = False
        self.steps = 0
        self.history = {
            'states': [self.state.copy()],
            'actions': [],
            'rewards': [],
            'score_diff': [0]
        }
        
        return self.state, {}
        
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (int): Action index
            
        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        # Unpack state
        score_diff, time_remaining, shot_clock, court_position, player_performance = self.state
        
        # Calculate success probability based on player performance and situation
        success_prob = self._calculate_success_probability(action, player_performance, court_position, shot_clock)
        
        # Determine if action is successful
        is_success = random.random() < success_prob
        
        # Calculate reward and update state based on action
        reward, score_change = self._calculate_reward_and_update(action, is_success, court_position)
        
        # Update score difference
        score_diff += score_change
        
        # Update time and shot clock
        time_change = self._calculate_time_change(action, is_success)
        time_remaining = max(0, time_remaining - time_change / 60.0)  # Convert seconds to minutes
        
        if action == 5:  # Timeout
            shot_clock = 24.0
        else:
            shot_clock = max(0, shot_clock - time_change)
            
        # If shot clock expires, turnover
        if shot_clock == 0:
            reward -= 2.0  # Penalty for shot clock violation
            shot_clock = 24.0
            court_position = random.uniform(0, 0.3)  # Move back to defensive end
            
        # If successful shot or turnover, reset shot clock
        if (action in [1, 2, 3] and is_success) or (action in [0, 1] and not is_success):
            shot_clock = 24.0
            
        # Update court position based on action and success
        court_position = self._update_court_position(action, is_success, court_position)
        
        # Update player performance with small random variation
        player_performance = min(1.0, max(0.0, player_performance + random.uniform(-0.05, 0.05)))
        
        # Update state
        self.state = np.array([score_diff, time_remaining, shot_clock, court_position, player_performance], dtype=np.float32)
        
        # Check if game is over
        self.steps += 1
        if time_remaining <= 0 or self.steps >= self.max_steps:
            self.done = True
            
            # Final reward based on game outcome
            if score_diff > 0:
                reward += 10.0  # Bonus for winning
            elif score_diff < 0:
                reward -= 10.0  # Penalty for losing
        
        # Update history
        self.history['states'].append(self.state.copy())
        self.history['actions'].append(action)
        self.history['rewards'].append(reward)
        self.history['score_diff'].append(score_diff)
        
        return self.state, reward, self.done, False, {"score_diff": score_diff, "success": is_success}
        
    def _calculate_success_probability(self, action, player_performance, court_position, shot_clock):
        """Calculate probability of action success based on state"""
        base_prob = self.base_success_prob[action]
        
        # Adjust for player performance
        performance_factor = 0.5 + player_performance / 2.0
        
        # Adjust for court position (except for timeout)
        position_factor = 1.0
        if action != 5:
            if action in [1, 2, 3]:  # Offensive actions
                position_factor = court_position  # Better near opponent basket
            else:
                position_factor = 1.0 - (court_position * 0.2)  # Slightly better in own half
        
        # Adjust for shot clock pressure
        clock_factor = 1.0
        if shot_clock < 5:
            clock_factor = 0.7  # Pressure reduces success probability
            
        return min(0.99, base_prob * performance_factor * position_factor * clock_factor)
        
    def _calculate_reward_and_update(self, action, is_success, court_position):
        """Calculate reward and score change based on action and result"""
        score_change = 0
        
        if action == 0:  # Pass
            if is_success:
                reward = 0.2  # Small positive reward for successful pass
            else:
                reward = -2.0  # Penalty for turnover
                
        elif action == 1:  # Drive
            if is_success:
                reward = 2.0
                score_change = 2
            else:
                if random.random() < 0.3:  # 30% chance of drawing foul
                    reward = 0.5
                    if random.random() < 0.7:  # 70% free throw success
                        score_change = 1
                        reward += 0.5
                    if random.random() < 0.7:  # 70% free throw success
                        score_change += 1
                        reward += 0.5
                else:
                    reward = -1.0  # Miss without foul
                    
        elif action == 2:  # 2-pointer
            if is_success:
                reward = 2.5
                score_change = 2
            else:
                reward = -0.5
                
        elif action == 3:  # 3-pointer
            if is_success:
                reward = 4.0
                score_change = 3
            else:
                reward = -1.0
                
        elif action == 4:  # Screen
            if is_success:
                reward = 0.5  # Creates advantage
            else:
                reward = -0.5  # Possible offensive foul
                
        elif action == 5:  # Timeout
            # Timeout value depends on court position and score situation
            if court_position < 0.3:  # Own half when calling timeout
                reward = -0.5  # Not ideal
            else:
                reward = 0.5  # Good to set up a play in offensive half
                
            # More valuable when losing
            if self.state[0] < -5:
                reward += 1.0
                
        return reward, score_change
        
    def _calculate_time_change(self, action, is_success):
        """Calculate time elapsed in seconds based on action"""
        if action == 0:  # Pass
            return random.uniform(2, 4)
        elif action == 1:  # Drive
            return random.uniform(3, 7)
        elif action == 2:  # 2-pointer
            return random.uniform(3, 8)
        elif action == 3:  # 3-pointer
            return random.uniform(3, 6)
        elif action == 4:  # Screen
            return random.uniform(4, 8)
        elif action == 5:  # Timeout
            return 0  # No game time consumed, but uses a timeout
            
    def _update_court_position(self, action, is_success, court_position):
        """Update court position based on action and result"""
        if action == 0:  # Pass
            if is_success:
                # Move slightly toward opponent basket
                return min(1.0, court_position + random.uniform(0.05, 0.2))
            else:
                # Turnover, move back to defensive end
                return random.uniform(0, 0.3)
                
        elif action in [1, 2, 3]:  # Shot attempts
            if is_success:
                # After successful shot, move back to defensive end
                return random.uniform(0, 0.3)
            else:
                # Rebound
                if random.random() < 0.3:  # 30% offensive rebound
                    return min(1.0, court_position + random.uniform(0, 0.1))
                else:
                    # Defensive rebound, move back
                    return random.uniform(0, 0.3)
                    
        elif action == 4:  # Screen
            # Slight movement toward opponent basket
            return min(1.0, court_position + random.uniform(0.02, 0.1))
            
        elif action == 5:  # Timeout
            # Position doesn't change
            return court_position
            
        return court_position  # Default fallback

class RLCoachAgent:
    """
    Class to handle RL agent training and evaluation for basketball decision-making.
    """
    def __init__(self):
        self.env = BasketballDecisionEnv()
        self.q_table = None
        
    def train_q_learning(self, total_timesteps=10000):
        """
        Train a Q-learning agent.
        
        Args:
            total_timesteps (int): Number of timesteps to train for
            
        Returns:
            dict: Q-table
        """
        # Initialize Q-table
        # Since we have continuous state space, this is a simplification
        # We'll discretize the state space for the Q-table
        state_bins = [20, 10, 8, 5, 5]  # Bins for each state dimension
        action_count = self.env.action_space.n
        
        # Initialize Q-table with zeros
        q_shape = state_bins + [action_count]
        self.q_table = np.zeros(q_shape)
        
        # Set learning parameters
        alpha = 0.1  # Learning rate
        gamma = 0.95  # Discount factor
        epsilon = 1.0  # Exploration rate
        epsilon_min = 0.01
        epsilon_decay = 0.995
        
        # Training loop
        obs, _ = self.env.reset()
        
        for t in range(total_timesteps):
            # Discretize state
            state_idx = self._discretize_state(obs, state_bins)
            
            # Choose action with epsilon-greedy policy
            if random.random() < epsilon:
                action = self.env.action_space.sample()  # Exploration
            else:
                action = np.argmax(self.q_table[tuple(state_idx)])  # Exploitation
            
            # Take action
            next_obs, reward, done, _, _ = self.env.step(action)
            
            # Discretize next state
            next_state_idx = self._discretize_state(next_obs, state_bins)
            
            # Update Q-value
            old_value = self.q_table[tuple(state_idx + [action])]
            next_max = np.max(self.q_table[tuple(next_state_idx)])
            
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            self.q_table[tuple(state_idx + [action])] = new_value
            
            # Update state
            obs = next_obs
            
            # Reset environment if done
            if done:
                obs, _ = self.env.reset()
                
            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            
            # Print progress
            if (t + 1) % 1000 == 0:
                print(f"Episode {t + 1}/{total_timesteps}")
        
        return self.q_table
        
    def _discretize_state(self, state, bins):
        """
        Discretize continuous state into bins for Q-table.
        
        Args:
            state (numpy.ndarray): Continuous state array
            bins (list): Number of bins for each state dimension
            
        Returns:
            list: Discretized state indices
        """
        # Define state bounds
        lower_bounds = self.env.observation_space.low
        upper_bounds = self.env.observation_space.high
        
        # Calculate discretized indices
        idx = []
        for s, lb, ub, b in zip(state, lower_bounds, upper_bounds, bins):
            # Scale state to [0, 1]
            scaled = (s - lb) / (ub - lb)
            # Convert to bin index
            bin_idx = min(b - 1, max(0, int(scaled * b)))
            idx.append(bin_idx)
            
        return idx
        
    def train_policy_gradient(self, total_timesteps=10000):
        """
        Train a policy gradient agent (simplified implementation).
        
        Args:
            total_timesteps (int): Number of timesteps to train for
            
        Returns:
            dict: Trained policy
        """
        # This is a placeholder - in a real implementation, this would use PPO or other policy gradient algorithm
        print("Policy gradient training would be implemented with stable_baselines3.PPO")
        
        # For now, just return the Q-learning strategy
        if self.q_table is None:
            self.train_q_learning(total_timesteps)
            
        return {"type": "q_table", "policy": self.q_table}
        
    def evaluate_agent(self, model, n_eval_episodes=10):
        """
        Evaluate a trained agent.
        
        Args:
            model: Trained RL model (Q-table in this simplified implementation)
            n_eval_episodes (int): Number of episodes to evaluate
            
        Returns:
            tuple: (mean_reward, std_reward)
        """
        # Reset environment
        obs, _ = self.env.reset()
        
        total_rewards = []
        episode_reward = 0
        
        # Evaluation loop
        for _ in range(n_eval_episodes):
            done = False
            episode_reward = 0
            steps = 0
            
            while not done and steps < 200:  # Limit steps to avoid infinite loops
                # Choose action from Q-table or policy
                if isinstance(model, np.ndarray):  # Q-table
                    state_idx = self._discretize_state(obs, [20, 10, 8, 5, 5])
                    action = np.argmax(model[tuple(state_idx)])
                else:  # Policy function
                    # This would be different for a real policy model
                    action = self.predict_action(model, obs)
                
                # Take action
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                steps += 1
                
            # Reset for next episode
            obs, _ = self.env.reset()
            total_rewards.append(episode_reward)
            
        # Calculate mean and std of rewards
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        return mean_reward, std_reward
        
    def predict_action(self, model, state):
        """
        Predict action from trained model (simplified).
        
        Args:
            model: Trained model (Q-table or policy)
            state (numpy.ndarray): Environment state
            
        Returns:
            int: Action to take
        """
        if isinstance(model, dict) and model.get("type") == "q_table":
            # Q-table prediction
            state_idx = self._discretize_state(state, [20, 10, 8, 5, 5])
            return np.argmax(model["policy"][tuple(state_idx)])
        elif isinstance(model, np.ndarray):
            # Direct Q-table
            state_idx = self._discretize_state(state, [20, 10, 8, 5, 5])
            return np.argmax(model[tuple(state_idx)])
        else:
            # Random action as fallback
            return self.env.action_space.sample()
            
    def get_action_probabilities(self, model, state):
        """
        Get action probabilities for a given state.
        
        Args:
            model: Trained RL model
            state (numpy.ndarray): Environment state
            
        Returns:
            numpy.ndarray: Action probabilities
        """
        action_count = self.env.action_space.n
        
        # This is a simplified implementation
        if isinstance(model, dict) and model.get("type") == "q_table":
            # Get Q-values
            state_idx = self._discretize_state(state, [20, 10, 8, 5, 5])
            q_values = model["policy"][tuple(state_idx)]
        elif isinstance(model, np.ndarray):
            # Direct Q-table
            state_idx = self._discretize_state(state, [20, 10, 8, 5, 5])
            q_values = model[tuple(state_idx)]
        else:
            # Uniform probabilities as fallback
            return np.ones(action_count) / action_count
            
        # Convert Q-values to probabilities with softmax
        q_exp = np.exp(q_values - np.max(q_values))  # Shift for numerical stability
        return q_exp / np.sum(q_exp)
        
    def simulate_game(self, model, initial_state=None, max_steps=50):
        """
        Simulate a game using a trained agent.
        
        Args:
            model: Trained RL model
            initial_state (numpy.ndarray): Initial state (optional)
            max_steps (int): Maximum number of steps to simulate
            
        Returns:
            dict: Game history
        """
        # Reset environment
        obs, _ = self.env.reset()
        
        # Set initial state if provided
        if initial_state is not None:
            # This would require modifying the environment's state directly
            # In a real implementation, this would be handled differently
            self.env.state = np.array(initial_state, dtype=np.float32)
            obs = self.env.state
            
        # Initialize tracking variables
        history = {
            'states': [obs.copy()],
            'actions': [],
            'rewards': [],
            'score_diff': [obs[0]],
            'action_names': ['Pass', 'Drive', 'Shoot 2PT', 'Shoot 3PT', 'Screen', 'Timeout'],
            'descriptions': []
        }
        
        # Simulation loop
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Choose action
            action = self.predict_action(model, obs)
            
            # Take action
            next_obs, reward, done, _, info = self.env.step(action)
            
            # Record event description
            success = info['success']
            description = self._generate_event_description(action, success, obs)
            
            # Update history
            history['states'].append(next_obs.copy())
            history['actions'].append(action)
            history['rewards'].append(reward)
            history['score_diff'].append(next_obs[0])
            history['descriptions'].append(description)
            
            # Update for next step
            obs = next_obs
            steps += 1
            
        return history
        
    def _generate_event_description(self, action, success, state):
        """Generate a description of the game event based on action and outcome"""
        # Unpack state variables
        score_diff, time_remaining, shot_clock, court_position, player_performance = state
        
        action_names = ['Passes', 'Drives to basket', 'Shoots 2-pointer', 'Shoots 3-pointer', 'Calls for screen', 'Calls timeout']
        
        zone = "offensive zone" if court_position > 0.5 else "backcourt"
        time_desc = f"with {int(shot_clock)} on shot clock"
        
        description = f"{action_names[action]} from {zone} {time_desc}"
        
        if action == 0:  # Pass
            if success:
                description += " - Complete"
            else:
                description += " - Turnover"
        elif action in [1, 2, 3]:  # Shot attempts
            if success:
                points = 2 if action == 1 or action == 2 else 3
                description += f" - Scores {points} points"
            else:
                description += " - Misses"
        elif action == 4:  # Screen
            if success:
                description += " - Creates space"
            else:
                description += " - Offensive foul"
        elif action == 5:  # Timeout
            description += " - Team regroups"
            
        return description
        
    def plot_game_simulation(self, history):
        """
        Plot game simulation results.
        
        Args:
            history (dict): Game history from simulate_game
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        # Create timeline of score difference
        steps = list(range(len(history['score_diff'])))
        time_remaining = [48 - (48 * step / max(1, len(steps) - 1)) for step in steps]
        
        # Create plot
        fig = go.Figure()
        
        # Add score difference line
        fig.add_trace(go.Scatter(
            x=time_remaining,
            y=history['score_diff'],
            mode='lines+markers',
            name='Score Difference',
            line=dict(color='blue', width=3),
            hovertext=history['descriptions'] if 'descriptions' in history else None
        ))
        
        # Add action markers
        if 'actions' in history and 'action_names' in history:
            for i, action in enumerate(history['actions']):
                action_name = history['action_names'][action]
                desc = history['descriptions'][i] if 'descriptions' in history else action_name
                
                fig.add_trace(go.Scatter(
                    x=[time_remaining[i+1]],
                    y=[history['score_diff'][i+1]],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'][action],
                        symbol=['circle', 'square', 'diamond', 'triangle-up', 'star', 'cross'][action]
                    ),
                    name=action_name,
                    text=desc,
                    hoverinfo='text',
                    showlegend=False
                ))
        
        # Layout
        fig.update_layout(
            title='Game Simulation',
            xaxis_title='Minutes Remaining',
            yaxis_title='Score Difference',
            hovermode='closest',
            legend=dict(
                x=0,
                y=1,
                traceorder='normal',
                font=dict(
                    family='sans-serif',
                    size=12,
                    color='black'
                ),
            )
        )
        
        # Add a zero line to show when the game is tied
        fig.add_shape(
            type='line',
            x0=min(time_remaining),
            y0=0,
            x1=max(time_remaining),
            y1=0,
            line=dict(
                color='rgba(0,0,0,0.3)',
                width=1,
                dash='dash'
            )
        )
        
        return fig
        
    def plot_action_distribution(self, history):
        """
        Plot distribution of actions taken during a game simulation.
        
        Args:
            history (dict): Game history from simulate_game
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if 'actions' not in history or 'action_names' not in history:
            return None
            
        # Count actions
        action_counts = {}
        for action in history['actions']:
            action_name = history['action_names'][action]
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
        # Create plot data
        actions = list(action_counts.keys())
        counts = list(action_counts.values())
        
        # Create bar plot
        fig = px.bar(
            x=actions,
            y=counts,
            title='Action Distribution',
            labels={'x': 'Action', 'y': 'Count'},
            color=actions,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_layout(
            xaxis_title='Action Type',
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
        
    def plot_reward_progression(self, history):
        """
        Plot the progression of rewards during a game simulation.
        
        Args:
            history (dict): Game history from simulate_game
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if 'rewards' not in history:
            return None
            
        # Create cumulative rewards
        steps = list(range(len(history['rewards'])))
        cumulative_rewards = np.cumsum(history['rewards'])
        
        # Create plot
        fig = px.line(
            x=steps,
            y=cumulative_rewards,
            title='Cumulative Reward Progression',
            labels={'x': 'Step', 'y': 'Cumulative Reward'}
        )
        
        fig.update_layout(
            xaxis_title='Simulation Step',
            yaxis_title='Cumulative Reward'
        )
        
        return fig
        
    def compare_models(self, models, n_simulations=5):
        """
        Compare different RL models through simulations.
        
        Args:
            models (dict): Dictionary of models to compare
            n_simulations (int): Number of simulations per model
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for name, model in models.items():
            model_results = {
                'final_scores': [],
                'mean_reward': 0,
                'win_rate': 0
            }
            
            total_reward = 0
            wins = 0
            
            for _ in range(n_simulations):
                # Run simulation
                history = self.simulate_game(model)
                
                # Record final score
                final_score = history['score_diff'][-1]
                model_results['final_scores'].append(final_score)
                
                # Calculate total reward
                total_reward += sum(history['rewards'])
                
                # Count win
                if final_score > 0:
                    wins += 1
                    
            # Calculate metrics
            model_results['mean_reward'] = total_reward / n_simulations
            model_results['win_rate'] = wins / n_simulations
            
            results[name] = model_results
            
        return results
        
    def plot_model_comparison(self, comparison_results):
        """
        Plot comparison of different models.
        
        Args:
            comparison_results (dict): Results from compare_models
            
        Returns:
            plotly.graph_objects.Figure: Plotly figure object
        """
        if not comparison_results:
            return None
            
        # Extract metrics
        models = list(comparison_results.keys())
        win_rates = [results['win_rate'] for _, results in comparison_results.items()]
        mean_rewards = [results['mean_reward'] for _, results in comparison_results.items()]
        
        # Create figure with two y-axes
        fig = go.Figure()
        
        # Add win rate bars
        fig.add_trace(go.Bar(
            x=models,
            y=win_rates,
            name='Win Rate',
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add mean reward line
        fig.add_trace(go.Scatter(
            x=models,
            y=mean_rewards,
            name='Mean Reward',
            mode='lines+markers',
            marker=dict(color='red', size=10),
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        # Update layout with second y-axis
        fig.update_layout(
            title='Model Comparison',
            xaxis_title='Model',
            yaxis=dict(
                title='Win Rate',
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue'),
                range=[0, 1]
            ),
            yaxis2=dict(
                title='Mean Reward',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            legend=dict(x=0.1, y=1.2, orientation='h')
        )
        
        return fig