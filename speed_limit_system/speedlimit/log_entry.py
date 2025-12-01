"""
Log entry module for the SpeedLimit System.

This module defines the LogEntry dataclass which represents a single log
record for a decision cycle. Log entries are used for traceability, debugging,
and auditing purposes, capturing the complete context of each speed limit
decision made by the system.
"""

from dataclasses import dataclass
from datetime import datetime

from .sensor_data import SensorData
from .speed_decision import SpeedDecision


@dataclass
class LogEntry:
    """
    Represents a single log record for a speed limit decision cycle.
    
    Contains all relevant information about a decision cycle, including the
    input sensor data, the resulting decision, and additional details about
    the decision-making process. This comprehensive logging enables system
    traceability, debugging, and post-analysis of system behavior.
    
    Attributes:
        timestamp: When the decision was made
        sensor_data: The sensor readings that were used as input
        decision: The final speed decision that was produced
        details: Additional metadata about the decision process (e.g., model
                 outputs, validation results, override information)
    """
    
    timestamp: datetime
    sensor_data: SensorData
    decision: SpeedDecision
    details: dict[str, str]
    
    def to_dict(self) -> dict[str, object]:
        """
        Converts the log entry to a serializable dictionary.
        
        This method is useful for storing log entries in databases, JSON files,
        or other serialization formats. The dictionary contains all log entry
        fields in a format that can be easily serialized.
        
        Returns:
            A dictionary representation of the log entry with all fields
            converted to serializable types
        """
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "sensor_data": {
                "illuminance": self.sensor_data.illuminance,
                "water_level": self.sensor_data.water_level,
                "temperature": self.sensor_data.temperature,
                "aqi": self.sensor_data.aqi,
                "timestamp": self.sensor_data.timestamp.isoformat() if self.sensor_data.timestamp else None,
            },
            "decision": {
                "final_speed": self.decision.final_speed,
                "decision_type": self.decision.decision_type,
            },
            "details": self.details,
        }

