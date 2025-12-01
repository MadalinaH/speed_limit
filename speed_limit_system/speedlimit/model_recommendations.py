"""
Model recommendations module for the SpeedLimit System.

This module defines the ModelRecommendations dataclass which aggregates speed
recommendations from both the Neural Network model and the LLM service, along
with validity flags indicating whether each recommendation should be used.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelRecommendations:
    """
    Aggregates speed recommendations from multiple models.
    
    Contains speed values and validity flags from both the Neural Network
    model and the LLM service. Provides helper methods to determine which
    recommendations are available for use in the decision-making process.
    
    Attributes:
        nn_speed: Speed recommendation from Neural Network model (km/h), or None
        nn_valid: Flag indicating if the Neural Network recommendation is valid
        llm_speed: Speed recommendation from LLM service (km/h), or None
        llm_valid: Flag indicating if the LLM recommendation is valid
    """
    
    nn_speed: Optional[int]
    nn_valid: bool
    llm_speed: Optional[int]
    llm_valid: bool
    
    def has_any_valid(self) -> bool:
        """
        Checks if at least one model has produced a valid recommendation.
        
        Returns:
            True if either nn_valid or llm_valid is True, False otherwise
        """
        return self.nn_valid or self.llm_valid
    
    def both_valid(self) -> bool:
        """
        Checks if both models have produced valid recommendations.
        
        Returns:
            True if both nn_valid and llm_valid are True, False otherwise
        """
        return self.nn_valid and self.llm_valid

