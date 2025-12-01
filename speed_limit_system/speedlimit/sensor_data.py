"""
Sensor data module for the SpeedLimit System.

This module defines the SensorData dataclass which represents sensor readings
from the environment, including weather conditions (illuminance, water level,
temperature) and air quality (AQI). It provides validation to ensure data
integrity before processing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class SensorData:
    """
    Represents sensor readings from the environment.
    
    Contains weather-related measurements (illuminance, water level, temperature)
    and air quality index (AQI). Optionally includes a timestamp for logging
    and tracking purposes.
    
    Attributes:
        illuminance: Light level measurement (must be >= 0)
        water_level: Water/precipitation level measurement (must be >= 0)
        temperature: Temperature in Celsius (must be between -40 and 60)
        aqi: Air Quality Index (must be >= 0)
        timestamp: Optional timestamp when the data was collected
    """
    
    illuminance: int
    water_level: int
    temperature: int
    aqi: int
    timestamp: Optional[datetime] = None
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        Validates all sensor data fields against acceptable ranges.
        
        Checks:
        - illuminance must be non-negative
        - water_level must be non-negative
        - temperature must be between -40 and 60 degrees Celsius
        - aqi must be non-negative
        
        Returns:
            A tuple containing:
            - bool: True if all validations pass, False otherwise
            - Optional[str]: None if valid, or a descriptive error message if invalid
        """
        # Check illuminance is non-negative
        if self.illuminance < 0:
            return (False, "illuminance must be >= 0")
        
        # Check water_level is non-negative
        if self.water_level < 0:
            return (False, "water_level must be >= 0")
        
        # Check temperature is within acceptable range (-40 to 60 Celsius)
        if self.temperature < -40 or self.temperature > 60:
            return (False, "temperature must be between -40 and 60")
        
        # Check aqi is non-negative
        if self.aqi < 0:
            return (False, "aqi must be >= 0")
        
        # All validations passed
        return (True, None)

