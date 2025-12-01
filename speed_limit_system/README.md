# Speed Limit Decision Support System

A decision support system that automatically determines safe speed limits based on sensor data, weather conditions, and air quality. The system combines neural network predictions and LLM recommendations to provide intelligent speed limit decisions.

## Features

- **Multi-Model Decision Making**: Combines neural network (NN) and LLM-based recommendations
- **Weather-Aware**: Detects hazardous weather conditions (low visibility, precipitation, black ice risk)
- **Air Quality Monitoring**: Integrates air quality index (AQI) for health-conscious speed recommendations
- **Web Interface**: Streamlit-based UI with three modes:
  - **Manual Input**: Test with custom sensor values
  - **Simulation Case**: Predefined hazard scenarios
  - **Live Mode**: Operational view with auto-updates and human override capability
- **Throttling**: Respects 30-second update intervals (REQ-9)
- **Persistent Logging**: Logs all decisions to `logs/system_log.log`

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd speed_limit_system
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Set up LLM service (for Grok integration):**
   
   The system works in "mock" mode by default (deterministic rule-based LLM behavior). To use real Grok LLM integration:
   
   - Get a Groq API key from [https://console.groq.com](https://console.groq.com)
   - Create a `.env` file in the project root:
     ```bash
     echo "GROQ_API_KEY=your_api_key_here" > .env
     ```
   - Or set the environment variable:
     ```bash
     export GROQ_API_KEY=your_api_key_here
     ```
   - Set the LLM mode:
     ```bash
     export SPEEDLIMIT_LLM_MODE=grok
     ```

## Running the Application

### Start the Web UI

Run the Streamlit application:

```bash
streamlit run ui/web_main.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Using the Web Interface

1. **Select a mode** from the sidebar:
   - **Manual input**: Use sliders to set sensor values manually
   - **Simulation case**: Choose from predefined scenarios (Normal, Dark & wet, Black ice risk, Poor air quality, Both hazards)
   - **Live mode**: System auto-updates every 30 seconds with realistic sensor data

2. **View results**:
   - Final speed recommendation (30-130 km/h)
   - Decision type (DEFAULT, NN_ONLY, LLM_ONLY, COMBINED, FALLBACK, OVERRIDE)
   - Active hazards (weather, air quality)
   - Detailed decision breakdown

3. **Live mode features**:
   - Auto-refresh every 30 seconds
   - Human override capability
   - Recent decisions log
   - Persistent logging to file

## Project Structure

```
speed_limit_system/
├── speedlimit/              # Core system modules
│   ├── speed_limit_system.py    # Main orchestrator
│   ├── sensor_data.py           # Sensor data validation
│   ├── weather_router.py        # Weather condition detection
│   ├── air_quality_router.py    # Air quality classification
│   ├── nn_model.py              # Neural network model
│   ├── llm_service.py           # LLM service (mock/grok)
│   ├── model_recommendations.py  # Model output combination
│   ├── speed_decision.py        # Speed decision data structure
│   ├── explanation.py            # Human-readable explanations
│   ├── log_entry.py             # Logging data structures
│   └── training/                # Training utilities
├── ui/                      # Web interface
│   └── web_main.py          # Streamlit application
├── tests/                   # Test suite
├── models/                  # Trained neural network models
├── data/                    # Training data
├── logs/                    # System logs
└── requirements.txt         # Python dependencies
```

## Running Tests

Run the test suite with pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_speed_limit_system.py

# Skip slow tests (e.g., real model tests)
pytest -m "not slow"

# Run with coverage report
pytest --cov=speedlimit --cov-report=html
```

## Configuration

### Environment Variables

- `GROQ_API_KEY`: Groq API key for LLM service (optional)
- `SPEEDLIMIT_LLM_MODE`: LLM mode selection (`mock` or `grok`, default: `mock`)

### Model Files

The system expects trained models in the `models/` directory:
- `nn_near_accident_risk.joblib`: Trained neural network model
- `nn_near_accident_risk_scaler.joblib`: Feature scaler
- `nn_near_accident_risk_features.txt`: Feature list

If models are missing, the system will use fallback behavior.

### Training Data

Training data should be located at `data/nn_training_data.csv`. The Live mode uses this dataset to sample sensor readings.

## System Behavior

### Speed Range
- **Minimum**: 30 km/h
- **Maximum**: 130 km/h
- **Default**: 130 km/h (normal conditions)
- **Fallback**: 80 km/h (on errors)

### Decision Types

- **DEFAULT**: Normal conditions, no hazards detected
- **NN_ONLY**: Only neural network provided valid recommendation
- **LLM_ONLY**: Only LLM service provided valid recommendation
- **COMBINED**: Both models provided recommendations (uses lower/more conservative speed)
- **FALLBACK**: System error or invalid inputs
- **OVERRIDE**: Human operator override (Live mode only)

### Update Throttling

The system enforces a minimum 30-second interval between computations (REQ-9). Repeated calls within this window return cached results.

## Troubleshooting

### Streamlit Not Starting

- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python3 --version` (should be 3.10+)
- Try clearing Streamlit cache: `streamlit cache clear`
