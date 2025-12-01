"""
Speed decision module for the SpeedLimit System.

This module defines the SpeedDecision dataclass which represents the final
speed recommendation produced by the SpeedLimitSystem. It includes the chosen
speed value and the type of decision that was made (e.g., based on model
recommendations, default, fallback, or human override).
"""

from dataclasses import dataclass


@dataclass
class SpeedDecision:
    """
    Represents the final speed decision made by the SpeedLimitSystem.
    
    Contains the chosen speed limit and metadata about how the decision was
    reached. The decision_type indicates which logic path was used to determine
    the final speed, which is useful for logging, debugging, and system
    transparency.
    
    Attributes:
        final_speed: The final speed limit in km/h (must be between 30 and 130)
        decision_type: Type of decision made, one of:
            - "DEFAULT": Normal conditions, using default highway speed (130 km/h)
            - "NN_ONLY": Using only Neural Network recommendation
            - "LLM_ONLY": Using only LLM service recommendation
            - "COMBINED": Using both models, taking the lower speed
            - "FALLBACK": Invalid inputs/outputs, using fallback speed (80 km/h)
            - "OVERRIDE": Human override was applied
    """
    
    final_speed: int
    decision_type: str
    
    def is_within_legal_range(self) -> bool:
        """
        Validates that the final speed is within the legal operating range.
        
        The system requires all speed decisions to be between 30 and 130 km/h
        for safety reasons. This method checks if the current decision meets
        this requirement.
        
        Returns:
            True if final_speed is between 30 and 130 (inclusive), False otherwise
        """
        return 30 <= self.final_speed <= 130

