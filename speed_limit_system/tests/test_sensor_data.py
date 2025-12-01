"""
Tests for SensorData component.

Tests cover:
- Equivalence classes: valid data, invalid data
- Boundary value analysis: validation thresholds
- Error scenarios: out-of-range values
- Full decision path coverage: all validation branches
"""

import pytest
from speedlimit.sensor_data import SensorData


class TestSensorDataValidation:
    """Test suite for SensorData validation."""
    
    # ==================== Equivalence Classes ====================
    
    def test_valid_sensor_data(self):
        """Equivalence class: All valid values → validation passes."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=100,
            temperature=15,
            aqi=50
        )
        valid, reason = sensor_data.validate()
        assert valid is True
        assert reason is None
    
    def test_invalid_illuminance_negative(self):
        """Error scenario: Negative illuminance → validation fails."""
        sensor_data = SensorData(
            illuminance=-1,  # Invalid
            water_level=100,
            temperature=15,
            aqi=50
        )
        valid, reason = sensor_data.validate()
        assert valid is False
        assert "illuminance" in reason.lower()
    
    def test_invalid_water_level_negative(self):
        """Error scenario: Negative water level → validation fails."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=-1,  # Invalid
            temperature=15,
            aqi=50
        )
        valid, reason = sensor_data.validate()
        assert valid is False
        assert "water_level" in reason.lower()
    
    def test_invalid_temperature_too_low(self):
        """Error scenario: Temperature too low (< -40) → validation fails."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=100,
            temperature=-50,  # Invalid (< -40)
            aqi=50
        )
        valid, reason = sensor_data.validate()
        assert valid is False
        assert "temperature" in reason.lower()
    
    def test_invalid_temperature_too_high(self):
        """Error scenario: Temperature too high (> 60) → validation fails."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=100,
            temperature=70,  # Invalid (> 60)
            aqi=50
        )
        valid, reason = sensor_data.validate()
        assert valid is False
        assert "temperature" in reason.lower()
    
    def test_invalid_aqi_negative(self):
        """Error scenario: Negative AQI → validation fails."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=100,
            temperature=15,
            aqi=-1  # Invalid
        )
        valid, reason = sensor_data.validate()
        assert valid is False
        assert "aqi" in reason.lower()
    
    # ==================== Boundary Value Analysis ====================
    
    def test_illuminance_at_min_boundary(self):
        """Boundary: Illuminance at minimum (0)."""
        sensor_data = SensorData(illuminance=0, water_level=100, temperature=15, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is True
    
    def test_illuminance_just_below_min(self):
        """Boundary: Illuminance just below minimum (-1)."""
        sensor_data = SensorData(illuminance=-1, water_level=100, temperature=15, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is False
    
    def test_water_level_at_min_boundary(self):
        """Boundary: Water level at minimum (0)."""
        sensor_data = SensorData(illuminance=1000, water_level=0, temperature=15, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is True
    
    def test_water_level_just_below_min(self):
        """Boundary: Water level just below minimum (-1)."""
        sensor_data = SensorData(illuminance=1000, water_level=-1, temperature=15, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is False
    
    def test_temperature_at_min_boundary(self):
        """Boundary: Temperature at minimum (-40)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=-40, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is True
    
    def test_temperature_just_below_min(self):
        """Boundary: Temperature just below minimum (-41)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=-41, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is False
    
    def test_temperature_at_max_boundary(self):
        """Boundary: Temperature at maximum (60)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=60, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is True
    
    def test_temperature_just_above_max(self):
        """Boundary: Temperature just above maximum (61)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=61, aqi=50)
        valid, _ = sensor_data.validate()
        assert valid is False
    
    def test_aqi_at_min_boundary(self):
        """Boundary: AQI at minimum (0)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=15, aqi=0)
        valid, _ = sensor_data.validate()
        assert valid is True
    
    def test_aqi_just_below_min(self):
        """Boundary: AQI just below minimum (-1)."""
        sensor_data = SensorData(illuminance=1000, water_level=100, temperature=15, aqi=-1)
        valid, _ = sensor_data.validate()
        assert valid is False
    
    # ==================== Decision Path Coverage ====================
    
    def test_validation_checks_all_fields(self):
        """Decision path: Validation checks all fields in order."""
        # Test that first invalid field is reported
        sensor_data = SensorData(illuminance=-1, water_level=-1, temperature=-50, aqi=-1)
        valid, reason = sensor_data.validate()
        assert valid is False
        # Should report first invalid field (illuminance)
        assert "illuminance" in reason.lower()
    
    def test_multiple_invalid_fields_reports_first(self):
        """Decision path: Multiple invalid fields → reports first encountered."""
        sensor_data = SensorData(illuminance=1000, water_level=-1, temperature=-50, aqi=-1)
        valid, reason = sensor_data.validate()
        assert valid is False
        # Should report water_level (first invalid after illuminance)
        assert "water_level" in reason.lower()

