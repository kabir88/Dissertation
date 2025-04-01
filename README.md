# NBA Strategy Optimization Platform

An AI-powered NBA analytics platform leveraging machine learning and reinforcement learning to analyze player performance and optimize coaching decisions.

## Features

- **Player Statistics Analysis**: Detailed performance metrics and visualizations for NBA players
- **Performance Prediction**: ML models to predict player stats using historical data
- **Player Clustering**: Identify similar player types and playstyles using K-means clustering
- **RL Decision Optimization**: Simulate game scenarios and optimize coaching decisions using reinforcement learning
- **Team Analysis**: Comprehensive team performance metrics and matchup predictions

## Technology Stack

- **Backend**: Python, MongoDB Atlas
- **AI/ML**: scikit-learn, Gymnasium (for RL)
- **Data**: NBA API for real-time stats
- **Visualization**: Streamlit, Plotly, Matplotlib
- **Deployment**: Render/Replit

## Setup Instructions

### Local Development

1. Clone the repository:
   ```
   git clone [your-repo-url]
   cd nba-strategy-app
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file with:
   ```
   MONGO_URI=your_mongodb_connection_string
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

### Deployment

#### Render

1. Connect your GitHub repository to Render
2. Create a new Web Service
3. Use the following settings:
   - Environment: Python
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add the environment variable `MONGO_URI`

## Project Structure

- `app.py`: Main Streamlit application entry point
- `pages/`: Additional Streamlit pages for different features
- `data_handler.py`: NBA API data fetching and processing
- `prediction_models.py`: Machine learning models for player stats prediction
- `clustering_models.py`: Player clustering and similarity analysis
- `rl_models.py`: Reinforcement learning models for decision optimization
- `backend/`: API and database utilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.