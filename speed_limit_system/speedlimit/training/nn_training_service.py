"""
Neural network training service module for the SpeedLimit System.

This module implements the training pipeline for the near-accident risk prediction
neural network. It loads training data, performs feature normalization, trains
an MLPClassifier, and saves the trained model for use in inference.

The model predicts near-accident risk based on:
- Environmental features: illuminance, water level, temperature
- Speed limit (candidate speed being evaluated)
- Other weather features if available

Model architecture:
- MLPClassifier (Multi-Layer Perceptron)
- 1-3 hidden layers
- Sigmoid output activation
- Binary cross-entropy loss
- Binary classification: near-accident (1) vs safe (0)
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
import joblib


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Prepares feature matrix and target vector from training data.
    
    Extracts relevant features:
    - L (illuminance/light)
    - W (water level)
    - T (temperature)
    - SpeedLimit (candidate speed)
    - Other weather features if available (H, AP, WS, WD, N, etc.)
    
    Args:
        df: Training DataFrame
    
    Returns:
        Tuple of (feature_matrix, target_vector, feature_names)
    """
    # Define feature columns (prioritize essential ones)
    essential_features = ['L', 'W', 'T', 'SpeedLimit']  # Light, Water, Temperature, Speed
    optional_features = ['H', 'AP', 'WS', 'WD', 'N']  # Humidity, Air Pressure, Wind Speed, Wind Direction, Noise
    
    # Select available features
    available_features = []
    for feat in essential_features + optional_features:
        if feat in df.columns:
            available_features.append(feat)
    
    if not available_features:
        raise ValueError("No feature columns found in training data")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df['near_accident'].copy()
    
    # Handle any remaining NaN values
    X = X.fillna(X.mean())
    
    return X, y, available_features


def train_and_save_model(data_path: str, model_path: str) -> None:
    """
    Trains the neural network model and saves it to disk.
    
    Training pipeline:
    1. Load training data
    2. Prepare features and target
    3. Split into training/validation sets (80/20)
    4. Normalize features using StandardScaler
    5. Train MLPClassifier with 1-3 hidden layers
    6. Evaluate on validation set
    7. Save model and scaler
    
    Args:
        data_path: Path to training data CSV file
        model_path: Path where trained model will be saved
    
    Raises:
        FileNotFoundError: If training data file doesn't exist
        ValueError: If training data is invalid
    """
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")
    
    print(f"Loading training data from: {data_path}")
    df = pd.read_csv(data_path_obj)
    print(f"Loaded {len(df)} training samples")
    
    # Prepare features
    print("Preparing features...")
    X, y, feature_names = _prepare_features(df)
    print(f"Features: {feature_names}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split into training and validation sets
    print("Splitting data into training/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    
    # Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train MLPClassifier
    print("Training MLPClassifier...")
    # Architecture: 2 hidden layers with 100 and 50 neurons
    # Activation: ReLU for hidden layers, sigmoid for output (via log_loss)
    # Solver: adam (adaptive moment estimation)
    # Learning rate: adaptive
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 hidden layers: 100 -> 50 neurons
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    print("Evaluating model...")
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    accuracy = accuracy_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    print(f"Validation Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    
    # Save model and scaler
    model_path_obj = Path(model_path)
    model_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = model_path_obj
    joblib.dump(model, model_file)
    print(f"Model saved to: {model_file}")
    
    # Save scaler (needed for inference)
    scaler_file = model_path_obj.parent / f"{model_path_obj.stem}_scaler.joblib"
    joblib.dump(scaler, scaler_file)
    print(f"Scaler saved to: {scaler_file}")
    
    # Save feature names (needed for inference)
    feature_names_file = model_path_obj.parent / f"{model_path_obj.stem}_features.txt"
    with open(feature_names_file, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"Feature names saved to: {feature_names_file}")


if __name__ == "__main__":
    # Default paths
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    
    data_path = project_root / "data" / "nn_training_data.csv"
    model_path = project_root / "models" / "nn_near_accident_risk.joblib"
    
    # Build training data if missing
    if not data_path.exists():
        print("Training data not found. Building training data...")
        from speedlimit.training.training_data_builder import build_training_data
        build_training_data(str(data_path))
    
    # Train and save model
    print("Training model...")
    train_and_save_model(str(data_path), str(model_path))
    print("Training complete!")

