"""
Weather router module for the SpeedLimit System.

This module contains the WeatherRouter class which analyzes sensor data to
determine if speed should be reduced due to weather conditions. It evaluates
conditions such as darkness (low illuminance) and black ice risk (combination
of high water level and freezing temperatures).
"""

from .sensor_data import SensorData


class WeatherRouter:
    """
    Router for weather-based speed reduction decisions.
    
    Analyzes sensor data to detect weather conditions that require speed
    reduction. Currently evaluates two conditions: darkness (low light levels)
    and black ice risk (freezing temperatures combined with high water levels).
    """
    
    def should_reduce_for_weather(self, sensor_data: SensorData) -> bool:
        """
        Determines if speed should be reduced due to weather conditions.
        
        Evaluates two weather conditions:
        - Darkness: illuminance below 500 (low light conditions)
        - Black ice: water level above 1000 AND temperature below 0°C
        
        Returns True if either condition is detected, indicating that speed
        reduction is needed for safety.
        
        Args:
            sensor_data: Sensor readings containing illuminance, water_level,
                        and temperature values
        
        Returns:
            True if speed reduction is needed due to weather conditions,
            False otherwise
        """
        # Check for dark conditions: low illuminance indicates poor visibility
        is_dark = sensor_data.illuminance < 500
        
        # Check for black ice conditions: freezing temperature with high water level
        # Black ice forms when water freezes on the road surface
        is_black_ice = sensor_data.water_level > 1000 and sensor_data.temperature < 0
        
        # Speed reduction needed if either dark conditions or black ice is detected
        return is_dark or is_black_ice

