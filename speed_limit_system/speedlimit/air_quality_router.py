"""
Air quality router module for the SpeedLimit System.

This module contains the AirQualityRouter class which is a pure classifier
for air quality conditions. It evaluates AQI (Air Quality Index) readings to
determine whether air quality is poor enough to require speed reduction, but
does not make speed recommendations itself.
"""

from .sensor_data import SensorData


class AirQualityRouter:
    """
    Pure classifier for air quality-based speed reduction decisions.
    
    Analyzes air quality index (AQI) values to determine if air quality is
    poor enough to warrant speed reduction. This is a classification-only
    component that does not make speed recommendations; those are handled by
    the LLMService when called by the SpeedLimitSystem orchestrator.
    """
    
    def is_poor_air_quality(self, sensor_data: SensorData) -> bool:
        """
        Determines if air quality is poor enough to require speed reduction.
        
        Uses a threshold of AQI > 100 to classify air quality as poor.
        Air quality indices above 100 are considered unhealthy and may require
        reduced speeds to minimize exposure and emissions.
        
        Args:
            sensor_data: Sensor readings containing the AQI value
        
        Returns:
            True if AQI > 100 (poor air quality), False otherwise
        """
        # Check if AQI exceeds the threshold for poor air quality
        return sensor_data.aqi > 100

