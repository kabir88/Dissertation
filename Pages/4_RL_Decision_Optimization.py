import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from rl_models import RLCoachAgent, BasketballDecisionEnv

st.title("Reinforcement Learning Decision Optimization")
st.write("Use AI to optimize in-game basketball decisions and strategy.")

# Initialize RL agent
rl_agent = RLCoachAgent()

# Sidebar options
st.sidebar.header("Model Options")
model_type = st.sidebar.radio(
    "Select Model Type",
    ["Q-Learning", "Policy Gradient"]
)

model_type_key = "q_learning" if model_type == "Q-Learning" else "policy_gradient"

# Training parameters
st.sidebar.subheader("Training Parameters")
training_steps = st.sidebar.slider("Training Steps", 1000, 50000, 10000, 1000)

# Game simulation options
st.sidebar.subheader("Game Simulation")
initial_score_diff = st.sidebar.slider("Initial Score Difference", -20, 20, 0, 1)
initial_time_remaining = st.sidebar.slider("Initial Time Remaining (minutes)", 1, 48, 10, 1)
initial_player_performance = st.sidebar.slider("Player Performance Rating", 0.5, 1.0, 0.7, 0.05)

# Train model and run simulation
if st.button("Train Model and Simulate Game"):
    with st.spinner(f"Training {model_type} model..."):
        # Train the selected model
        if model_type == "Q-Learning":
            model = rl_agent.train_q_learning(total_timesteps=training_steps)
            st.success("Q-Learning model trained successfully!")
        else:
            model = rl_agent.train_policy_gradient(total_timesteps=training_steps)
            st.success("Policy Gradient model trained successfully!")
        
        # Evaluate the model
        mean_reward, std_reward = rl_agent.evaluate_agent(model, n_eval_episodes=10)
        
        st.info(f"Model Evaluation: Average Reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Set up initial state for simulation
        initial_state = np.array([
            initial_score_diff,          # Score difference
            initial_time_remaining,      # Time remaining
            24.0,                        # Shot clock
            0.5,                         # Court position
            initial_player_performance   # Player performance
        ], dtype=np.float32)
        
        # Run simulation
        st.header("Game Simulation")
        with st.spinner("Simulating game with trained model..."):
            simulation_history = rl_agent.simulate_game(
                model, 
                initial_state=initial_state,
                max_steps=50
            )
            
            # Game summary
            final_score_diff = simulation_history['score_diffs'][-1]
            final_time = simulation_history['time_remaining'][-1]
            total_reward = sum(simulation_history['rewards'])
            
            outcome = "Win" if final_score_diff > 0 else "Loss" if final_score_diff < 0 else "Tie"
            
            # Display metrics
            metrics_cols = st.columns(3)
            metrics_cols[0].metric("Final Score Differential", f"{final_score_diff:.1f}")
            metrics_cols[1].metric("Game Outcome", outcome)
            metrics_cols[2].metric("Total Reward", f"{total_reward:.1f}")
            
            # Plot simulation
            st.subheader("Game Progression")
            fig_game = rl_agent.plot_game_simulation(simulation_history)
            st.plotly_chart(fig_game, use_container_width=True)
            
            # Plot action distribution
            st.subheader("Decision Distribution")
            fig_actions = rl_agent.plot_action_distribution(simulation_history)
            st.plotly_chart(fig_actions, use_container_width=True)
            
            # Plot reward progression
            st.subheader("Reward Progression")
            fig_rewards = rl_agent.plot_reward_progression(simulation_history)
            st.plotly_chart(fig_rewards, use_container_width=True)
            
            # Show detailed decision sequence
            st.subheader("Decision Sequence")
            
            # Convert the action sequence to action names
            action_names = simulation_history['action_names']
            actions = simulation_history['actions']
            states = simulation_history['states']
            rewards = simulation_history['rewards']
            score_diffs = simulation_history['score_diffs']
            time_remaining = simulation_history['time_remaining']
            
            # Create a dataframe for the decision sequence
            decision_data = []
            for i in range(len(actions)):
                decision_data.append({
                    "Step": i + 1,
                    "Action": action_names[actions[i]],
                    "Score Diff": score_diffs[i+1],
                    "Time Left": time_remaining[i+1],
                    "Reward": rewards[i],
                    "Court Position": states[i+1][3],
                    "Shot Clock": states[i+1][2]
                })
            
            decision_df = pd.DataFrame(decision_data)
            st.dataframe(decision_df)
            
            # Decision analysis
            st.subheader("Decision Analysis")
            
            # Analyze decision patterns
            if len(actions) > 0:
                # Most frequent action
                action_counts = {}
                for a in actions:
                    action_name = action_names[a]
                    if action_name in action_counts:
                        action_counts[action_name] += 1
                    else:
                        action_counts[action_name] = 1
                
                most_freq_action = max(action_counts, key=action_counts.get)
                
                # Best actions (highest rewards)
                action_rewards = {}
                for a, r in zip(actions, rewards):
                    action_name = action_names[a]
                    if action_name in action_rewards:
                        action_rewards[action_name].append(r)
                    else:
                        action_rewards[action_name] = [r]
                
                avg_rewards = {a: sum(rs)/len(rs) if rs else 0 for a, rs in action_rewards.items()}
                best_action = max(avg_rewards, key=avg_rewards.get) if avg_rewards else None
                
                # Display analysis
                analysis_cols = st.columns(2)
                with analysis_cols[0]:
                    st.write("**Most Frequent Decision:**")
                    st.write(f"{most_freq_action} ({action_counts[most_freq_action]} times)")
                    
                    # Frequency chart
                    fig_freq = px.pie(
                        names=list(action_counts.keys()),
                        values=list(action_counts.values()),
                        title="Decision Frequency"
                    )
                    st.plotly_chart(fig_freq, use_container_width=True)
                    
                with analysis_cols[1]:
                    if best_action:
                        st.write("**Most Rewarding Decision:**")
                        st.write(f"{best_action} (Avg. reward: {avg_rewards[best_action]:.2f})")
                        
                        # Reward by action chart
                        reward_df = pd.DataFrame({
                            'Action': list(avg_rewards.keys()),
                            'Average Reward': list(avg_rewards.values())
                        })
                        
                        fig_rewards_by_action = px.bar(
                            reward_df,
                            x='Action',
                            y='Average Reward',
                            title="Average Reward by Decision",
                            color='Average Reward',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_rewards_by_action, use_container_width=True)
                
                # Situational analysis
                st.subheader("Situational Decision Analysis")
                
                # Decisions when ahead vs behind
                ahead_actions = [action_names[actions[i]] for i in range(len(actions)) 
                                if score_diffs[i] > 0]
                behind_actions = [action_names[actions[i]] for i in range(len(actions)) 
                                 if score_diffs[i] < 0]
                
                # Count actions
                ahead_counts = {}
                for a in ahead_actions:
                    if a in ahead_counts:
                        ahead_counts[a] += 1
                    else:
                        ahead_counts[a] = 1
                
                behind_counts = {}
                for a in behind_actions:
                    if a in behind_counts:
                        behind_counts[a] += 1
                    else:
                        behind_counts[a] = 1
                
                # Create dataframes
                situation_data = []
                
                # Add ahead data
                for action, count in ahead_counts.items():
                    situation_data.append({
                        'Action': action,
                        'Count': count,
                        'Situation': 'Ahead'
                    })
                
                # Add behind data
                for action, count in behind_counts.items():
                    situation_data.append({
                        'Action': action,
                        'Count': count,
                        'Situation': 'Behind'
                    })
                
                if situation_data:
                    situation_df = pd.DataFrame(situation_data)
                    
                    # Plot
                    fig_situation = px.bar(
                        situation_df,
                        x='Action',
                        y='Count',
                        color='Situation',
                        barmode='group',
                        title="Decisions by Game Situation"
                    )
                    st.plotly_chart(fig_situation, use_container_width=True)
                    
                    # Written analysis
                    st.write("**Key Insights:**")
                    
                    if ahead_actions:
                        most_common_ahead = max(ahead_counts, key=ahead_counts.get)
                        st.write(f"- When ahead, the AI mostly chose to {most_common_ahead}")
                    
                    if behind_actions:
                        most_common_behind = max(behind_counts, key=behind_counts.get)
                        st.write(f"- When behind, the AI mostly chose to {most_common_behind}")
                    
                    # Time management
                    late_game_actions = [action_names[actions[i]] for i in range(len(actions)) 
                                       if time_remaining[i] < 5]
                    
                    if late_game_actions:
                        late_counts = {}
                        for a in late_game_actions:
                            if a in late_counts:
                                late_counts[a] += 1
                            else:
                                late_counts[a] = 1
                        
                        most_common_late = max(late_counts, key=late_counts.get)
                        st.write(f"- In the last 5 minutes, the AI mostly chose to {most_common_late}")
                    
            else:
                st.write("No decisions were made in the simulation.")

# Explain the reinforcement learning approach
with st.expander("How Reinforcement Learning Works in Basketball"):
    st.write("""
    ### Reinforcement Learning for Basketball Decision Making
    
    Reinforcement Learning (RL) is an AI approach where an agent learns to make decisions by taking actions in an environment to maximize cumulative rewards. In the basketball context:
    
    1. **Environment**: The basketball game state (score difference, time remaining, court position, etc.)
    2. **Agent**: The AI coach/player making decisions
    3. **Actions**: Basketball decisions (pass, shoot, drive, etc.)
    4. **Rewards**: Points scored, strategic advantages, game outcomes
    
    ### Models Used:
    
    - **Q-Learning**: Learns the value of taking specific actions in specific states. It creates a "Q-table" mapping state-action pairs to expected rewards.
    
    - **Policy Gradient**: Directly learns a policy (strategy) that maps states to the best actions without using a value function intermediate.
    
    ### Applications:
    
    - **In-Game Decision Making**: Optimize decisions based on game state
    - **Strategy Development**: Discover effective strategies for different scenarios
    - **Player Development**: Identify optimal usage of player skills
    
    The simulation above shows how an AI agent makes decisions in a simplified basketball environment to maximize the team's chances of winning.
    """)

# Reference information
st.sidebar.markdown("---")
st.sidebar.subheader("Action Reference")
st.sidebar.markdown("""
- **Pass**: Move the ball, low risk
- **Drive**: Attack the basket
- **Shoot 2PT**: Attempt a 2-point shot
- **Shoot 3PT**: Attempt a 3-point shot
- **Screen**: Call for a screen
- **Timeout**: Use a timeout
""")