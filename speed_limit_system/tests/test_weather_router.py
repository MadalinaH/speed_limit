"""
Tests for WeatherRouter component.

Tests cover:
- Equivalence classes: safe conditions, hazardous conditions
- Boundary value analysis: thresholds for dark (500) and black ice (water>1000, temp<0)
- Error scenarios: edge cases and invalid inputs
- Full decision path coverage: all branches in should_reduce_for_weather
"""

import pytest
from speedlimit.weather_router import WeatherRouter
from speedlimit.sensor_data import SensorData


class TestWeatherRouter:
    """Test suite for WeatherRouter component."""
    
    @pytest.fixture
    def weather_router(self):
        """Fixture providing a WeatherRouter instance."""
        return WeatherRouter()
    
    # ==================== Equivalence Classes ====================
    
    def test_safe_conditions_no_reduction(self, weather_router):
        """Equivalence class: Clearly safe weather conditions."""
        # Good visibility, no water, normal temperature
        sensor_data = SensorData(
            illuminance=1000,  # Well above dark threshold (500)
            water_level=0,     # No water
            temperature=15,   # Above freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_dark_conditions_reduction(self, weather_router):
        """Equivalence class: Dark conditions trigger reduction."""
        # Low illuminance (dark)
        sensor_data = SensorData(
            illuminance=300,  # Below dark threshold (500)
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_black_ice_conditions_reduction(self, weather_router):
        """Equivalence class: Black ice conditions trigger reduction."""
        # Freezing temperature with high water level
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1500,  # Above black ice threshold (1000)
            temperature=-5,    # Below freezing (0)
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_both_hazards_reduction(self, weather_router):
        """Equivalence class: Both dark and black ice trigger reduction."""
        # Dark AND black ice conditions
        sensor_data = SensorData(
            illuminance=300,   # Dark
            water_level=1500,  # High water
            temperature=-5,   # Freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    # ==================== Boundary Value Analysis ====================
    
    def test_illuminance_boundary_below_threshold(self, weather_router):
        """Boundary: illuminance just below dark threshold (499)."""
        sensor_data = SensorData(
            illuminance=499,  # Just below 500
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_illuminance_boundary_at_threshold(self, weather_router):
        """Boundary: illuminance exactly at dark threshold (500)."""
        sensor_data = SensorData(
            illuminance=500,  # Exactly at threshold
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_illuminance_boundary_above_threshold(self, weather_router):
        """Boundary: illuminance just above dark threshold (501)."""
        sensor_data = SensorData(
            illuminance=501,  # Just above 500
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_water_level_boundary_below_threshold(self, weather_router):
        """Boundary: water level just below black ice threshold (999)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=999,  # Just below 1000
            temperature=-5,   # Freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_water_level_boundary_at_threshold(self, weather_router):
        """Boundary: water level exactly at black ice threshold (1000)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1000,  # Exactly at threshold
            temperature=-5,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False  # Not > 1000
    
    def test_water_level_boundary_above_threshold(self, weather_router):
        """Boundary: water level just above black ice threshold (1001)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1001,  # Just above 1000
            temperature=-5,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_temperature_boundary_below_freezing(self, weather_router):
        """Boundary: temperature just below freezing (-1)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1500,
            temperature=-1,  # Just below 0
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_temperature_boundary_at_freezing(self, weather_router):
        """Boundary: temperature exactly at freezing (0)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1500,
            temperature=0,  # Exactly at freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False  # Not < 0
    
    def test_temperature_boundary_above_freezing(self, weather_router):
        """Boundary: temperature just above freezing (1)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1500,
            temperature=1,  # Just above 0
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_black_ice_requires_both_conditions(self, weather_router):
        """Boundary: Black ice requires BOTH high water AND freezing temp."""
        # High water but not freezing
        sensor_data = SensorData(
            illuminance=1000,
            water_level=1500,  # High water
            temperature=5,     # Not freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
        
        # Freezing but no water
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,     # No water
            temperature=-5,    # Freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    # ==================== Edge Cases ====================
    
    def test_zero_illuminance_dark(self, weather_router):
        """Edge case: Zero illuminance (complete darkness)."""
        sensor_data = SensorData(
            illuminance=0,
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_very_high_illuminance_safe(self, weather_router):
        """Edge case: Very high illuminance (bright daylight)."""
        sensor_data = SensorData(
            illuminance=100000,
            water_level=0,
            temperature=15,
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_very_high_water_level_not_freezing(self, weather_router):
        """Edge case: Very high water but not freezing (no black ice)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=10000,  # Very high water
            temperature=10,     # Not freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    def test_extreme_cold_no_water(self, weather_router):
        """Edge case: Extreme cold but no water (no black ice)."""
        sensor_data = SensorData(
            illuminance=1000,
            water_level=0,
            temperature=-30,  # Very cold
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False
    
    # ==================== Decision Path Coverage ====================
    
    def test_dark_only_path(self, weather_router):
        """Decision path: Only dark condition triggers (no black ice)."""
        sensor_data = SensorData(
            illuminance=400,  # Dark
            water_level=500,  # Low water (not > 1000)
            temperature=10,   # Not freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_black_ice_only_path(self, weather_router):
        """Decision path: Only black ice triggers (not dark)."""
        sensor_data = SensorData(
            illuminance=600,  # Not dark (>= 500)
            water_level=1500,  # High water
            temperature=-5,   # Freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is True
    
    def test_neither_condition_path(self, weather_router):
        """Decision path: Neither condition triggers (no reduction)."""
        sensor_data = SensorData(
            illuminance=600,  # Not dark
            water_level=500,  # Low water
            temperature=10,   # Not freezing
            aqi=50
        )
        assert weather_router.should_reduce_for_weather(sensor_data) is False

