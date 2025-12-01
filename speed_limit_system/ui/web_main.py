"""
Web UI module for the SpeedLimit System.

This module provides a Streamlit-based web interface for interacting with the
SpeedLimit system. Supports three modes: Manual input (testing), Simulation case
(demo scenarios), and Live mode (operational view with periodic updates and override).
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Literal
import random

# Add project root to Python path to enable imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np

from speedlimit.sensor_data import SensorData
from speedlimit.weather_router import WeatherRouter
from speedlimit.air_quality_router import AirQualityRouter
from speedlimit.nn_model import NeuralNetworkModel
from speedlimit.llm_service import LLMService
from speedlimit.speed_limit_system import SpeedLimitSystem
from speedlimit.speed_decision import SpeedDecision
from speedlimit.explanation import Explanation
from speedlimit.log_entry import LogEntry


# Initialize session state for live mode logging and system instance
if "live_mode_log" not in st.session_state:
    st.session_state.live_mode_log: List[Dict[str, Any]] = []

# Initialize or recreate system instance if it's missing required attributes or has wrong cache version
if ("speed_limit_system" not in st.session_state or 
    not hasattr(st.session_state.speed_limit_system, 'clear_cache') or
    not hasattr(st.session_state.speed_limit_system, '_last_sensor_data_hash') or
    not hasattr(st.session_state.speed_limit_system, 'CACHE_VERSION') or
    getattr(st.session_state.speed_limit_system, 'CACHE_VERSION', 0) != SpeedLimitSystem.CACHE_VERSION):
    st.session_state.speed_limit_system = SpeedLimitSystem()

# Initialize training dataset cache
if "training_dataset_cache" not in st.session_state:
    st.session_state.training_dataset_cache = None


def load_training_dataset() -> Optional[pd.DataFrame]:
    """
    Loads the training dataset and caches it in session state.
    
    Returns:
        DataFrame with training data, or None if loading fails
    """
    # Return cached dataset if available
    if st.session_state.training_dataset_cache is not None:
        return st.session_state.training_dataset_cache
    
    # Try to load from data/nn_training_data.csv
    training_data_path = project_root / "data" / "nn_training_data.csv"
    
    if not training_data_path.exists():
        return None
    
    try:
        df = pd.read_csv(training_data_path)
        # Cache the dataset
        st.session_state.training_dataset_cache = df
        return df
    except Exception:
        return None


def sample_sensor_data_from_training(
    profile: Literal["random", "good_weather", "bad_weather"] = "random"
) -> SensorData:
    """
    Samples a random row from the training dataset and converts it to SensorData.
    
    If the dataset cannot be loaded, falls back to generating synthetic values.
    
    Args:
        profile: Sampling profile mode:
            - "random": Random row from dataset
            - "good_weather": Bias towards rows with good weather (high illuminance, low water)
            - "bad_weather": Bias towards rows with bad weather (low illuminance, high water)
    
    Returns:
        SensorData object with sampled or synthetic values
    """
    df = load_training_dataset()
    
    if df is None or len(df) == 0:
        # Fallback: Generate synthetic sensor values
        return SensorData(
            illuminance=random.randint(500, 1500),
            water_level=random.randint(0, 1000),
            temperature=random.randint(-5, 25),
            aqi=random.randint(30, 100)
        )
    
    # Apply profile-based filtering if needed
    filtered_df = df.copy()
    
    if profile == "good_weather":
        # Filter for good weather conditions: high illuminance (L), low water (W)
        if 'L' in filtered_df.columns and 'W' in filtered_df.columns:
            # Take rows with L > median and W < median
            l_median = filtered_df['L'].median()
            w_median = filtered_df['W'].median()
            filtered_df = filtered_df[
                (filtered_df['L'] > l_median) & (filtered_df['W'] < w_median)
            ]
            # If no rows match, fall back to random
            if len(filtered_df) == 0:
                filtered_df = df.copy()
    
    elif profile == "bad_weather":
        # Filter for bad weather conditions: low illuminance (L), high water (W)
        if 'L' in filtered_df.columns and 'W' in filtered_df.columns:
            # Take rows with L < median and W > median
            l_median = filtered_df['L'].median()
            w_median = filtered_df['W'].median()
            filtered_df = filtered_df[
                (filtered_df['L'] < l_median) & (filtered_df['W'] > w_median)
            ]
            # If no rows match, fall back to random
            if len(filtered_df) == 0:
                filtered_df = df.copy()
    
    # Sample a random row
    sampled_row = filtered_df.sample(n=1).iloc[0]
    
    # Map columns to SensorData fields
    # L = Light (illuminance in lux) -> convert to millilux (multiply by 1000)
    # W = Water level (already in micrometers)
    # T = Temperature (already in Celsius)
    illuminance = int(sampled_row.get('L', 500) * 1000) if pd.notna(sampled_row.get('L')) else 500
    water_level = int(sampled_row.get('W', 0)) if pd.notna(sampled_row.get('W')) else 0
    temperature = int(round(sampled_row.get('T', 5))) if pd.notna(sampled_row.get('T')) else 5
    
    # Synthesize AQI (not in dataset) using realistic distribution
    # Most AQI values are moderate (50-100), with occasional high values
    aqi_roll = random.random()
    if aqi_roll < 0.7:  # 70% moderate
        aqi = random.randint(30, 100)
    elif aqi_roll < 0.9:  # 20% good
        aqi = random.randint(0, 50)
    else:  # 10% poor
        aqi = random.randint(100, 200)
    
    # Ensure values are within valid ranges
    illuminance = max(0, illuminance)
    water_level = max(0, water_level)
    temperature = max(-40, min(60, temperature))
    aqi = max(0, aqi)
    
    return SensorData(
        illuminance=illuminance,
        water_level=water_level,
        temperature=temperature,
        aqi=aqi
    )


def main() -> None:
    """
    Main function that runs the Streamlit web interface.
    
    Sets up the page layout, handles three modes (Manual input, Simulation case,
    Live mode), processes decisions through the SpeedLimitSystem, and displays
    results with appropriate controls for each mode.
    """
    # Set page title
    st.set_page_config(page_title="Speed Limit Decision Support", layout="wide")
    st.title("Speed Limit Decision Support")
    
    # Mode selection in sidebar
    mode = st.sidebar.selectbox(
        "Mode",
        ["Manual input", "Simulation case", "Live mode"],
        help="Choose mode: Manual input (testing), Simulation case (demo), or Live mode (operational)"
    )
    
    # Clear cache button in sidebar (useful for debugging or after code changes)
    if st.sidebar.button("Clear Cache", help="Clear computation cache to force fresh calculation"):
        if "speed_limit_system" in st.session_state:
            # Check if method exists (in case of old cached instance)
            if hasattr(st.session_state.speed_limit_system, 'clear_cache'):
                st.session_state.speed_limit_system.clear_cache()
            else:
                # Recreate instance if it's an old version without clear_cache
                st.session_state.speed_limit_system = SpeedLimitSystem()
        st.sidebar.success("Cache cleared")
    
    # Live mode: Enable auto-refresh every 30 seconds (matches backend throttling)
    if mode == "Live mode":
        st_autorefresh(interval=30000, limit=None, key="live_refresh")  # 30 seconds = 30000 ms
        st.sidebar.info("🔄 Live mode active: Updates every 30 seconds")
    
    # Two-column layout
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.header("Input Parameters")
        
        # Get sensor data based on selected mode
        if mode == "Manual input":
            # Manual input mode: sliders for each sensor value
            illuminance = st.slider(
                "Illuminance (millilux)",
                min_value=0,
                max_value=2000,
                value=500,
                help="Light level measurement"
            )
            water_level = st.slider(
                "Water level (µm)",
                min_value=0,
                max_value=3000,
                value=0,
                help="Water/precipitation level"
            )
            temperature = st.slider(
                "Temperature (°C)",
                min_value=-20,
                max_value=40,
                value=5,
                help="Temperature in Celsius"
            )
            aqi = st.slider(
                "AQI",
                min_value=0,
                max_value=300,
                value=50,
                help="Air Quality Index"
            )
        elif mode == "Simulation case":
            # Simulation mode: predefined scenarios
            scenario = st.selectbox(
                "Select scenario",
                [
                    "Normal conditions",
                    "Dark & wet",
                    "Black ice risk",
                    "Poor air quality",
                    "Both hazards"
                ]
            )
            
            # Define scenario values
            scenarios = {
                "Normal conditions": {"illum": 1200, "water": 0, "temp": 15, "aqi": 50},
                "Dark & wet": {"illum": 300, "water": 800, "temp": 5, "aqi": 70},
                "Black ice risk": {"illum": 600, "water": 2000, "temp": -3, "aqi": 80},
                "Poor air quality": {"illum": 1000, "water": 0, "temp": 20, "aqi": 160},
                "Both hazards": {"illum": 300, "water": 1800, "temp": -2, "aqi": 170}
            }
            
            selected = scenarios[scenario]
            illuminance = selected["illum"]
            water_level = selected["water"]
            temperature = selected["temp"]
            aqi = selected["aqi"]
            
            # Display selected scenario values
            st.caption("Scenario values:")
            st.text(f"Illuminance: {illuminance} millilux")
            st.text(f"Water level: {water_level} µm")
            st.text(f"Temperature: {temperature} °C")
            st.text(f"AQI: {aqi}")
        else:  # Live mode
            # Live mode: Sample from training dataset (no manual input)
            st.info("📡 Live mode: System auto-updates every 30 seconds using realistic sensor data from the provided dataset.")
            
            # Profile selection for sampling
            profile = st.selectbox(
                "Sampling profile",
                ["random", "good_weather", "bad_weather"],
                help="Profile for sampling: random (any conditions), good_weather (bias towards clear conditions), bad_weather (bias towards hazardous conditions)",
                key="live_profile_select"
            )
            
            # Track previous profile to detect changes
            if "previous_live_profile" not in st.session_state:
                st.session_state.previous_live_profile = profile
            
            # Track last computation time to detect auto-refresh cycles
            if "last_live_computation_time" not in st.session_state:
                st.session_state.last_live_computation_time = datetime.now()
            
            # Check if this is an auto-refresh cycle (30+ seconds since last computation)
            time_since_last = (datetime.now() - st.session_state.last_live_computation_time).total_seconds()
            is_auto_refresh = time_since_last >= 29.0  # Allow slight margin for timing
            
            # Check if profile changed
            profile_changed = st.session_state.previous_live_profile != profile
            if profile_changed:
                st.session_state.previous_live_profile = profile
                is_auto_refresh = True  # Force refresh on profile change
            
            # "Jump to new random conditions" button
            jump_button_key = "jump_to_new_conditions"
            if st.button("🔄 Jump to new random conditions", help="Immediately sample a new random sensor reading", key=jump_button_key):
                # Force new sample on button click
                is_auto_refresh = True  # Force refresh on button click
            
            # Initialize or update sensor data
            if "live_sensor_data" not in st.session_state or is_auto_refresh:
                st.session_state.live_sensor_data = sample_sensor_data_from_training(profile)
                if is_auto_refresh:
                    st.session_state.last_live_computation_time = datetime.now()
            
            # Get current sensor data
            current_sensor_data = st.session_state.live_sensor_data
            
            # Display sampled sensor values (read-only)
            st.subheader("Current Sensor Readings (Auto-generated)")
            st.write(f"**Illuminance:** {current_sensor_data.illuminance} millilux")
            st.write(f"**Water level:** {current_sensor_data.water_level} µm")
            st.write(f"**Temperature:** {current_sensor_data.temperature} °C")
            st.write(f"**AQI:** {current_sensor_data.aqi}")
            st.caption("These values are automatically sampled from the training dataset.")
            
            # Extract values for computation
            illuminance = current_sensor_data.illuminance
            water_level = current_sensor_data.water_level
            temperature = current_sensor_data.temperature
            aqi = current_sensor_data.aqi
        
        # Create SensorData from inputs
        sensor_data = SensorData(
            illuminance=illuminance,
            water_level=water_level,
            temperature=temperature,
            aqi=aqi
        )
        
        # Instantiate system components (shared across all modes)
        weather_router = WeatherRouter()
        air_router = AirQualityRouter()
        nn_model = NeuralNetworkModel()
        llm_service = LLMService()
        # Use session state to persist system instance for throttling
        system = st.session_state.speed_limit_system
        
        # Compute speed decision (respects 30-second throttling in backend)
        # Enable persistent logging only in Live mode
        enable_logging = (mode == "Live mode")
        automatic_decision, explanation, log_entry = system.compute_speed(
            sensor_data,
            weather_router,
            air_router,
            nn_model,
            llm_service,
            enable_persistent_logging=enable_logging
        )
        
        # Human override section (only in Live mode)
        final_decision = automatic_decision
        if mode == "Live mode":
            st.divider()
            st.subheader("Human Override")
            apply_override = st.checkbox("Apply human override")
            override_speed = st.number_input(
                "Override speed (km/h)",
                min_value=30,
                max_value=130,
                value=automatic_decision.final_speed,
                help="Override the system's decision (must be between 30 and 130 km/h)"
            )
            
            # Apply override if requested
            override_applied = False
            if apply_override:
                overridden_decision = system.apply_override(automatic_decision, override_speed)
                if overridden_decision.decision_type == "OVERRIDE":
                    final_decision = overridden_decision
                    override_applied = True
                else:
                    st.warning("Override rejected: speed must be between 30 and 130 km/h")
        
        # Live mode: Log decisions (automatic and overrides)
        if mode == "Live mode":
            # Extract hazard flags
            weather_flag_str = log_entry.details.get("weather_reduction_needed", "False")
            poor_air_str = log_entry.details.get("poor_air_quality", "False")
            weather_flag = weather_flag_str.lower() == "true" if isinstance(weather_flag_str, str) else bool(weather_flag_str)
            poor_air = poor_air_str.lower() == "true" if isinstance(poor_air_str, str) else bool(poor_air_str)
            
            # Log automatic decision (only if this is a new decision, not a cached one)
            log_entry_dict = {
                "timestamp": log_entry.timestamp,
                "automatic_speed": automatic_decision.final_speed,
                "decision_type": automatic_decision.decision_type,
                "weather_hazard": weather_flag,
                "poor_air": poor_air
            }
            
            # Only add automatic decision if timestamp is different from last entry (avoid duplicates from refresh)
            if (not st.session_state.live_mode_log or 
                st.session_state.live_mode_log[-1]["timestamp"] != log_entry.timestamp):
                st.session_state.live_mode_log.append(log_entry_dict)
                # Keep only last 20 entries
                if len(st.session_state.live_mode_log) > 20:
                    st.session_state.live_mode_log.pop(0)
            
            # Log override if applied (use current timestamp for override events)
            if override_applied:
                override_log_entry = {
                    "timestamp": datetime.now(),
                    "automatic_speed": automatic_decision.final_speed,
                    "decision_type": "OVERRIDE",
                    "final_speed": final_decision.final_speed,
                    "weather_hazard": weather_flag,
                    "poor_air": poor_air
                }
                st.session_state.live_mode_log.append(override_log_entry)
                # Keep only last 20 entries
                if len(st.session_state.live_mode_log) > 20:
                    st.session_state.live_mode_log.pop(0)
    
    with right_col:
        st.header("Decision Output")
        
        # Extract hazard flags from log entry details (normalize boolean strings)
        weather_flag_str = log_entry.details.get("weather_reduction_needed", "False")
        poor_air_str = log_entry.details.get("poor_air_quality", "False")
        weather_flag = weather_flag_str.lower() == "true" if isinstance(weather_flag_str, str) else bool(weather_flag_str)
        poor_air = poor_air_str.lower() == "true" if isinstance(poor_air_str, str) else bool(poor_air_str)
        
        # Display final speed as metric
        st.metric("🚗 Final speed (km/h)", final_decision.final_speed)
        
        # Display decision type
        st.caption(f"Decision type: {final_decision.decision_type}")
        
        # Show automatic recommendation if override was applied
        if final_decision.decision_type == "OVERRIDE":
            st.caption(
                f"Automatic recommendation: {automatic_decision.final_speed} km/h "
                f"({automatic_decision.decision_type})"
            )
        
        # Construct hazard text
        hazard_parts = []
        if weather_flag:
            hazard_parts.append("Weather risk")
        if poor_air:
            hazard_parts.append("Poor air quality")
        hazard_text = ", ".join(hazard_parts) if hazard_parts else "None"
        
        # Display hazard status
        st.caption(f"⚠️ Active hazards: {hazard_text}")
        
        # Display decision status message
        decision_type = final_decision.decision_type
        if decision_type in ["DEFAULT", "NN_ONLY", "LLM_ONLY", "COMBINED"]:
            st.success("Automatic decision based on current conditions.")
        elif decision_type == "OVERRIDE":
            st.warning("Human override is active. Final speed differs from the automatic recommendation.")
        elif decision_type == "FALLBACK":
            st.error("Fallback speed used due to invalid input or model failure.")
        
        # Structured decision breakdown
        with st.expander("🧠 Decision breakdown", expanded=False):
            st.write(f"- Weather hazard detected: {weather_flag}")
            st.write(f"- Poor air quality detected: {poor_air}")
            
            # Extract model recommendations from log entry
            nn_speed_str = log_entry.details.get("nn_speed", "None")
            llm_speed_str = log_entry.details.get("llm_speed", "None")
            nn_valid_str = log_entry.details.get("nn_valid", "False")
            llm_valid_str = log_entry.details.get("llm_valid", "False")
            
            nn_speed_display = nn_speed_str if nn_speed_str != "None" else "None (not applicable)"
            llm_speed_display = llm_speed_str if llm_speed_str != "None" else "None (not applicable)"
            
            st.write(f"- NN recommended speed: {nn_speed_display} km/h")
            st.write(f"- LLM recommended speed: {llm_speed_display} km/h")
            
            # Determine which rule was applied
            nn_valid = nn_valid_str.lower() == "true" if isinstance(nn_valid_str, str) else bool(nn_valid_str)
            llm_valid = llm_valid_str.lower() == "true" if isinstance(llm_valid_str, str) else bool(llm_valid_str)
            
            if nn_valid and llm_valid:
                rule_text = "Minimum of valid NN and LLM speeds (both exist)"
            elif nn_valid:
                rule_text = "NN recommendation only (LLM not applicable)"
            elif llm_valid:
                rule_text = "LLM recommendation only (NN not applicable)"
            elif not weather_flag and not poor_air:
                rule_text = "Default speed (normal conditions, no model outputs)"
            else:
                rule_text = "Fallback speed (abnormal conditions, no valid model outputs)"
            
            st.write(f"- Rule applied: {rule_text}")
            st.write(f"- Automatic speed (before override): {automatic_decision.final_speed} km/h")
            
            if final_decision.decision_type == "OVERRIDE":
                st.write(f"- Human override applied: {final_decision.final_speed} km/h")
            
            # Show natural language explanation
            st.info(explanation.text)
        
        # Speed range visualization (updated to 30-130 km/h)
        st.caption("Speed range: 30–130 km/h")
        normalized = (final_decision.final_speed - 30) / (130 - 30)
        normalized = min(max(normalized, 0.0), 1.0)
        st.progress(normalized)
        
        # Live mode: Show recent decisions table
        if mode == "Live mode" and st.session_state.live_mode_log:
            st.divider()
            st.subheader("Recent Decisions (Live Mode)")
            
            # Prepare table data
            table_data = []
            for entry in reversed(st.session_state.live_mode_log[-10:]):  # Show last 10
                hazards = []
                if entry["weather_hazard"]:
                    hazards.append("Weather")
                if entry["poor_air"]:
                    hazards.append("Air")
                hazards_str = ", ".join(hazards) if hazards else "None"
                
                # Handle override entries which have final_speed instead of just automatic_speed
                if entry["decision_type"] == "OVERRIDE" and "final_speed" in entry:
                    speed_display = f"{entry['final_speed']} (override, auto: {entry['automatic_speed']})"
                else:
                    speed_display = entry["automatic_speed"]
                
                table_data.append({
                    "Timestamp": entry["timestamp"].strftime("%H:%M:%S"),
                    "Speed (km/h)": speed_display,
                    "Decision Type": entry["decision_type"],
                    "Hazards": hazards_str
                })
            
            if table_data:
                st.dataframe(table_data, use_container_width=True, height=300)
                st.caption(f"Showing last {len(table_data)} decisions. Updates respect 30-second throttling.")
        
        # Display additional details in expander
        with st.expander("View detailed log entry"):
            st.json({
                "timestamp": log_entry.timestamp.isoformat(),
                "sensor_data": {
                    "illuminance": log_entry.sensor_data.illuminance,
                    "water_level": log_entry.sensor_data.water_level,
                    "temperature": log_entry.sensor_data.temperature,
                    "aqi": log_entry.sensor_data.aqi
                },
                "decision": {
                    "final_speed": log_entry.decision.final_speed,
                    "decision_type": log_entry.decision.decision_type
                },
                "details": log_entry.details
            })


if __name__ == "__main__":
    main()
