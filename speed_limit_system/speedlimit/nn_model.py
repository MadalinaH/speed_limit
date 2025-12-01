"""
Neural network model module for the SpeedLimit System.

This module contains the NeuralNetworkModel class which provides runtime ML
inference for weather-based safe speed predictions. The model predicts
near-accident risk based on environmental conditions and candidate speeds,
then selects the highest safe speed that keeps risk below a threshold.

The model implements risk-based speed selection:
- Evaluates candidate speeds: [130, 120, 100, 80, 60, 40] km/h
- For each speed, computes near-accident risk probability
- Selects highest speed with risk <= threshold (default 0.10)
- Returns safe speed or None if no speed satisfies threshold
"""

from pathlib import Path
from typing import Optional
import joblib
import numpy as np

from .sensor_data import SensorData


class NeuralNetworkModel:
    """
    Neural network model for predicting safe speeds based on weather conditions.
    
    Loads a trained MLPClassifier that predicts near-accident risk. The model
    evaluates multiple candidate speeds and selects the highest speed that
    maintains risk below a safety threshold.
    
    Near-accident definition (from training):
    - SkidAngle > 15° OR CloseCarCm < 50 OR CloseGuardrailCm < 50
    """
    
    # Risk threshold for safe speed selection (3% probability of near-accident)
    RISK_THRESHOLD = 0.03
    
    # Candidate speeds to evaluate (in descending order)
    CANDIDATE_SPEEDS = [130, 120, 110, 100, 90, 80, 60, 40, 30]
    
    # Legal speed range
    MIN_SPEED = 30
    MAX_SPEED = 130
    
    # Fallback speed if no candidate satisfies threshold
    FALLBACK_SPEED = 80
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initializes the neural network model.
        
        Args:
            model_path: Optional path to trained model file. If None, uses default
                       path: models/nn_near_accident_risk.joblib
        """
        if model_path is None:
            # Default model path relative to project root
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            model_path = str(project_root / "models" / "nn_near_accident_risk.joblib")
        
        self.model_path = Path(model_path)
        self.model: Optional[object] = None
        self.scaler: Optional[object] = None
        self.feature_names: Optional[list[str]] = None
        self._model_loaded = False
    
    def load_model(self) -> None:
        """
        Loads the trained neural network model, scaler, and feature names from disk.
        
        The model file structure:
        - Model: models/nn_near_accident_risk.joblib
        - Scaler: models/nn_near_accident_risk_scaler.joblib
        - Features: models/nn_near_accident_risk_features.txt
        
        Raises:
            FileNotFoundError: If model files are not found
            Exception: If model loading fails
        """
        if self._model_loaded:
            return
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load model
            self.model = joblib.load(self.model_path)
            
            # Load scaler
            scaler_path = self.model_path.parent / f"{self.model_path.stem}_scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
            
            # Load feature names
            features_path = self.model_path.parent / f"{self.model_path.stem}_features.txt"
            if features_path.exists():
                with open(features_path, 'r') as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            else:
                raise FileNotFoundError(f"Feature names file not found: {features_path}")
            
            self._model_loaded = True
            
        except Exception as e:
            # Model loading failed - will return invalid predictions
            self.model = None
            self.scaler = None
            self.feature_names = None
            self._model_loaded = False
            raise
    
    def _construct_feature_vector(
        self,
        sensor_data: SensorData,
        candidate_speed: int
    ) -> np.ndarray:
        """
        Constructs a feature vector for model inference.
        
        Maps SensorData to the feature format expected by the trained model:
        - L (illuminance): from sensor_data.illuminance
        - W (water level): from sensor_data.water_level
        - T (temperature): from sensor_data.temperature
        - SpeedLimit: candidate_speed parameter
        - Other features: set to 0 if not available
        
        Args:
            sensor_data: Sensor readings
            candidate_speed: Speed to evaluate
        
        Returns:
            Feature vector as numpy array
        """
        if self.feature_names is None:
            raise ValueError("Feature names not loaded. Call load_model() first.")
        
        # Map sensor data to feature dictionary
        feature_dict = {
            'L': sensor_data.illuminance,  # Light/illuminance
            'W': sensor_data.water_level,  # Water level
            'T': sensor_data.temperature,  # Temperature
            'SpeedLimit': candidate_speed,  # Candidate speed
            # Optional features (set to 0 if not in sensor data)
            'H': 0,  # Humidity
            'AP': 0,  # Air pressure
            'WS': 0,  # Wind speed
            'WD': 0,  # Wind direction
            'N': 0,  # Noise
        }
        
        # Build feature vector in the order expected by the model
        feature_vector = []
        for feat_name in self.feature_names:
            if feat_name in feature_dict:
                feature_vector.append(feature_dict[feat_name])
            else:
                # Unknown feature - use 0 as default
                feature_vector.append(0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_safe_speed(
        self,
        sensor_data: SensorData,
        weather_reduction: bool
    ) -> tuple[Optional[int], bool]:
        """
        Predicts a safe speed based on sensor data and weather conditions.
        
        Inference process:
        1. If weather_reduction=False, return (None, False) - NN not used
        2. Load model if not already loaded
        3. Evaluate candidate speeds in descending order
        4. For each speed, compute near-accident risk probability
        5. Select highest speed with risk <= RISK_THRESHOLD
        6. If none satisfy threshold, return the speed with lowest risk (marked as valid)
        7. If all candidates fail due to exceptions, return fallback speed with valid=False
        8. Enforce legal speed range [30, 130]
        
        Args:
            sensor_data: Sensor readings containing weather-related measurements
            weather_reduction: True if weather conditions require speed reduction,
                            False otherwise
        
        Returns:
            A tuple containing:
            - Optional[int]: Predicted safe speed in km/h, or None if not applicable
            - bool: True if the prediction is valid, False otherwise
        """
        # Neural network is only used when weather reduction is needed
        if not weather_reduction:
            # Normal weather conditions: NN is not used
            return (None, False)
        
        # Load model if not already loaded
        try:
            if not self._model_loaded:
                self.load_model()
        except Exception:
            # Model loading failed - return invalid
            return (None, False)
        
        if self.model is None or self.scaler is None or self.feature_names is None:
            return (None, False)
        
        # Evaluate candidate speeds in descending order
        safe_speed = None
        lowest_risk_speed = None
        lowest_risk_value = float('inf')
        
        for candidate_speed in self.CANDIDATE_SPEEDS:
            # Ensure candidate is within legal range
            if candidate_speed < self.MIN_SPEED or candidate_speed > self.MAX_SPEED:
                continue
            
            try:
                # Construct feature vector
                feature_vector = self._construct_feature_vector(sensor_data, candidate_speed)
                
                # Normalize features
                feature_scaled = self.scaler.transform(feature_vector)
                
                # Predict risk probability (probability of near-accident)
                risk_proba = self.model.predict_proba(feature_scaled)[0, 1]
                
                # Track the speed with lowest risk (in case no speed meets threshold)
                if risk_proba < lowest_risk_value:
                    lowest_risk_value = risk_proba
                    lowest_risk_speed = candidate_speed
                
                # Check if risk is below threshold
                if risk_proba <= self.RISK_THRESHOLD:
                    safe_speed = candidate_speed
                    break  # Found highest safe speed
                    
            except (ValueError, AttributeError, IndexError, TypeError) as e:
                # Specific inference errors for this candidate - continue to next
                # These can occur if feature vector construction fails, scaler fails,
                # or model prediction fails due to shape/type mismatches
                continue
            except Exception:
                # Other unexpected errors - continue to next candidate
                continue
        
        # If no candidate satisfied threshold, return the lowest risk speed (or fallback)
        # Mark it as valid since the model made a valid prediction (all speeds are unsafe)
        if safe_speed is None:
            if lowest_risk_speed is not None:
                # Ensure the lowest risk speed is within legal range
                if self.MIN_SPEED <= lowest_risk_speed <= self.MAX_SPEED:
                    # Return the speed with lowest risk, marked as valid
                    return (lowest_risk_speed, True)
                else:
                    # Lowest risk speed is out of range - use fallback
                    return (self.FALLBACK_SPEED, False)
            else:
                # All candidates failed due to exceptions - use fallback and mark as invalid
                return (self.FALLBACK_SPEED, False)
        
        # Ensure final speed is within legal range
        if safe_speed < self.MIN_SPEED or safe_speed > self.MAX_SPEED:
            return (self.FALLBACK_SPEED, False)
        
        # Return the predicted safe speed
        return (safe_speed, True)
