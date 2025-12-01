"""
Training module for the SpeedLimit System.

This module contains components for building training data and training
the neural network model for near-accident risk prediction.
"""

from .training_data_builder import build_training_data
from .nn_training_service import train_and_save_model

__all__ = ['build_training_data', 'train_and_save_model']

