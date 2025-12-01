"""
Tests for AirQualityRouter component.

Tests cover:
- Equivalence classes: good, moderate, unhealthy AQI ranges
- Boundary value analysis: AQI threshold at 100
- Error scenarios: edge cases
- Full decision path coverage: all branches in is_poor_air_quality
"""

import pytest
from speedlimit.air_quality_router import AirQualityRouter
from speedlimit.sensor_data import SensorData


class TestAirQualityRouter:
    """Test suite for AirQualityRouter component."""
    
    @pytest.fixture
    def air_router(self):
        """Fixture providing an AirQualityRouter instance."""
        return AirQualityRouter()
    
    # ==================== Equivalence Classes ====================
    
    def test_good_air_quality_no_reduction(self, air_router):
        """Equivalence class: Good air quality (AQI <= 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=50  # Good air quality
        )
        assert air_router.is_poor_air_quality(sensor_data) is False
    
    def test_moderate_air_quality_no_reduction(self, air_router):
        """Equivalence class: Moderate air quality (AQI = 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=100  # At threshold
        )
        assert air_router.is_poor_air_quality(sensor_data) is False
    
    def test_unhealthy_air_quality_reduction(self, air_router):
        """Equivalence class: Unhealthy air quality (AQI > 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=150  # Unhealthy
        )
        assert air_router.is_poor_air_quality(sensor_data) is True
    
    def test_very_unhealthy_air_quality_reduction(self, air_router):
        """Equivalence class: Very unhealthy air quality (AQI >> 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=300  # Very unhealthy
        )
        assert air_router.is_poor_air_quality(sensor_data) is True
    
    # ==================== Boundary Value Analysis ====================
    
    def test_aqi_boundary_below_threshold(self, air_router):
        """Boundary: AQI just below threshold (99)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=99  # Just below 100
        )
        assert air_router.is_poor_air_quality(sensor_data) is False
    
    def test_aqi_boundary_at_threshold(self, air_router):
        """Boundary: AQI exactly at threshold (100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=100  # Exactly at threshold
        )
        assert air_router.is_poor_air_quality(sensor_data) is False  # Not > 100
    
    def test_aqi_boundary_above_threshold(self, air_router):
        """Boundary: AQI just above threshold (101)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=101  # Just above 100
        )
        assert air_router.is_poor_air_quality(sensor_data) is True
    
    # ==================== Edge Cases ====================
    
    def test_zero_aqi_good_air(self, air_router):
        """Edge case: Zero AQI (perfect air quality)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=0
        )
        assert air_router.is_poor_air_quality(sensor_data) is False
    
    def test_very_high_aqi_poor_air(self, air_router):
        """Edge case: Very high AQI (hazardous air quality)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=500  # Hazardous
        )
        assert air_router.is_poor_air_quality(sensor_data) is True
    
    # ==================== Decision Path Coverage ====================
    
    def test_poor_air_path(self, air_router):
        """Decision path: Poor air quality detected (AQI > 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=150
        )
        assert air_router.is_poor_air_quality(sensor_data) is True
    
    def test_good_air_path(self, air_router):
        """Decision path: Good air quality (AQI <= 100)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=80
        )
        assert air_router.is_poor_air_quality(sensor_data) is False

