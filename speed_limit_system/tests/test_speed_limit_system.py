"""
Tests for SpeedLimitSystem orchestrator component.

Tests cover:
- Equivalence classes: no hazards, weather only, air only, both hazards
- Boundary value analysis: speed limits, model outputs
- Error scenarios: invalid inputs, model failures, out-of-range speeds
- Full decision path coverage: all decision types (DEFAULT, NN_ONLY, LLM_ONLY, COMBINED, FALLBACK, OVERRIDE)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
from speedlimit.speed_limit_system import SpeedLimitSystem
from speedlimit.sensor_data import SensorData
from speedlimit.weather_router import WeatherRouter
from speedlimit.air_quality_router import AirQualityRouter
from speedlimit.nn_model import NeuralNetworkModel
from speedlimit.llm_service import LLMService
from speedlimit.speed_decision import SpeedDecision


class TestSpeedLimitSystem:
    """Test suite for SpeedLimitSystem orchestrator."""
    
    @pytest.fixture
    def system(self):
        """Fixture providing a SpeedLimitSystem instance."""
        return SpeedLimitSystem()
    
    @pytest.fixture
    def weather_router(self):
        """Fixture providing a WeatherRouter instance."""
        return WeatherRouter()
    
    @pytest.fixture
    def air_router(self):
        """Fixture providing an AirQualityRouter instance."""
        return AirQualityRouter()
    
    @pytest.fixture
    def nn_model(self):
        """Fixture providing a mocked NeuralNetworkModel."""
        model = Mock(spec=NeuralNetworkModel)
        return model
    
    @pytest.fixture
    def llm_service(self):
        """Fixture providing a mocked LLMService."""
        service = Mock(spec=LLMService)
        return service
    
    @pytest.fixture
    def valid_sensor_data(self):
        """Fixture providing valid sensor data."""
        return SensorData(
            illuminance=1000,
            water_level=0,
            temperature=15,
            aqi=50
        )
    
    # ==================== Equivalence Classes ====================
    
    def test_no_hazards_default_speed(self, system, weather_router, air_router, nn_model, llm_service, valid_sensor_data):
        """Equivalence class: No hazards → DEFAULT speed (130 km/h)."""
        # Mock: no weather reduction, no poor air
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, explanation, log_entry = system.compute_speed(
            valid_sensor_data, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 130
        assert decision.decision_type == "DEFAULT"
        assert "Normal conditions" in explanation.text
    
    def test_weather_hazard_only_nn_recommendation(self, system, weather_router, air_router, nn_model, llm_service):
        """Equivalence class: Weather hazard only → NN_ONLY."""
        sensor_data = SensorData(
            illuminance=300,  # Dark (weather hazard)
            water_level=0,
            temperature=15,
            aqi=50  # Good air
        )
        
        nn_model.predict_safe_speed.return_value = (80, True)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, explanation, log_entry = system.compute_speed(
            sensor_data, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 80
        assert decision.decision_type == "NN_ONLY"
        assert "Neural network recommended" in explanation.text
    
    def test_air_quality_hazard_only_llm_recommendation(self, system, weather_router, air_router, nn_model, llm_service):
        """Equivalence class: Air quality hazard only → LLM_ONLY."""
        sensor_data = SensorData(
            illuminance=1000,  # Not dark
            water_level=0,
            temperature=15,
            aqi=150  # Poor air
        )
        
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (60, True)
        
        decision, explanation, log_entry = system.compute_speed(
            sensor_data, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 60
        assert decision.decision_type == "LLM_ONLY"
        assert "LLM service recommended" in explanation.text
    
    def test_both_hazards_combined_recommendation(self, system, weather_router, air_router, nn_model, llm_service):
        """Equivalence class: Both hazards → COMBINED (min of both)."""
        sensor_data = SensorData(
            illuminance=300,  # Dark
            water_level=0,
            temperature=15,
            aqi=150  # Poor air
        )
        
        nn_model.predict_safe_speed.return_value = (80, True)
        llm_service.get_recommended_speed.return_value = (60, True)
        
        decision, explanation, log_entry = system.compute_speed(
            sensor_data, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 60  # Min of 80 and 60
        assert decision.decision_type == "COMBINED"
        assert "Both models recommended" in explanation.text
    
    def test_both_hazards_models_fail_fallback(self, system, weather_router, air_router, nn_model, llm_service):
        """Equivalence class: Hazards present but models fail → FALLBACK."""
        sensor_data = SensorData(
            illuminance=300,  # Dark
            water_level=0,
            temperature=15,
            aqi=150  # Poor air
        )
        
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, explanation, log_entry = system.compute_speed(
            sensor_data, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 80  # Fallback speed
        assert decision.decision_type == "FALLBACK"
        assert "fallback speed" in explanation.text.lower()
    
    # ==================== Boundary Value Analysis ====================
    
    def test_nn_speed_at_min_boundary(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: NN speed at minimum (30 km/h)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (30, True)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 30
    
    def test_nn_speed_at_max_boundary(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: NN speed at maximum (130 km/h)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (130, True)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 130
    
    def test_nn_speed_below_min_rejected(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: NN speed below minimum (29) → rejected, uses fallback."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (29, True)  # Below min
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 80  # Fallback
        assert decision.decision_type == "FALLBACK"
    
    def test_nn_speed_above_max_rejected(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: NN speed above maximum (131) → rejected, uses fallback."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (131, True)  # Above max
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 80  # Fallback
        assert decision.decision_type == "FALLBACK"
    
    def test_combined_speeds_nn_lower(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: Combined mode, NN speed is lower."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (60, True)
        llm_service.get_recommended_speed.return_value = (80, True)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 60  # Min of 60 and 80
    
    def test_combined_speeds_llm_lower(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: Combined mode, LLM speed is lower."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (80, True)
        llm_service.get_recommended_speed.return_value = (60, True)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 60  # Min of 80 and 60
    
    def test_combined_speeds_equal(self, system, weather_router, air_router, nn_model, llm_service):
        """Boundary: Combined mode, both speeds equal."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (70, True)
        llm_service.get_recommended_speed.return_value = (70, True)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.final_speed == 70
    
    # ==================== Error Scenarios ====================
    
    def test_invalid_sensor_data_fallback(self, system, weather_router, air_router, nn_model, llm_service):
        """Error scenario: Invalid sensor data → FALLBACK."""
        invalid_sensor = SensorData(
            illuminance=-1,  # Invalid (negative)
            water_level=0,
            temperature=15,
            aqi=50
        )
        
        decision, explanation, log_entry = system.compute_speed(
            invalid_sensor, weather_router, air_router, nn_model, llm_service
        )
        
        assert decision.final_speed == 80
        assert decision.decision_type == "FALLBACK"
        assert "invalid" in explanation.text.lower()
    
    def test_nn_model_failure_fallback(self, system, weather_router, air_router, nn_model, llm_service):
        """Error scenario: NN model returns invalid → handled gracefully."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        # Model returns invalid result (not exception, as predict_safe_speed handles exceptions internally)
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        # Should use fallback since weather hazard but no valid model output
        assert decision.decision_type == "FALLBACK"
        assert decision.final_speed == 80
    
    def test_llm_service_failure_fallback(self, system, weather_router, air_router, nn_model, llm_service):
        """Error scenario: LLM service returns invalid → handled gracefully."""
        sensor_data = SensorData(illuminance=1000, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (None, False)
        # LLM service handles exceptions internally and returns (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        # Should use fallback since poor air but no valid model output
        assert decision.decision_type == "FALLBACK"
        assert decision.final_speed == 80
    
    # ==================== Decision Path Coverage ====================
    
    def test_decision_path_default(self, system, weather_router, air_router, nn_model, llm_service, valid_sensor_data):
        """Decision path: DEFAULT (no hazards, no valid models)."""
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(valid_sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "DEFAULT"
        assert decision.final_speed == 130
    
    def test_decision_path_nn_only(self, system, weather_router, air_router, nn_model, llm_service):
        """Decision path: NN_ONLY (weather hazard, only NN valid)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (70, True)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "NN_ONLY"
        assert decision.final_speed == 70
    
    def test_decision_path_llm_only(self, system, weather_router, air_router, nn_model, llm_service):
        """Decision path: LLM_ONLY (poor air, only LLM valid)."""
        sensor_data = SensorData(illuminance=1000, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (65, True)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "LLM_ONLY"
        assert decision.final_speed == 65
    
    def test_decision_path_combined(self, system, weather_router, air_router, nn_model, llm_service):
        """Decision path: COMBINED (both hazards, both models valid)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (75, True)
        llm_service.get_recommended_speed.return_value = (60, True)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "COMBINED"
        assert decision.final_speed == 60
    
    def test_decision_path_fallback_hazards_no_models(self, system, weather_router, air_router, nn_model, llm_service):
        """Decision path: FALLBACK (hazards present but models fail)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=150)
        nn_model.predict_safe_speed.return_value = (None, False)
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "FALLBACK"
        assert decision.final_speed == 80
    
    def test_decision_path_fallback_invalid_output(self, system, weather_router, air_router, nn_model, llm_service):
        """Decision path: FALLBACK (model output out of range)."""
        sensor_data = SensorData(illuminance=300, water_level=0, temperature=15, aqi=50)
        nn_model.predict_safe_speed.return_value = (200, True)  # Out of range
        llm_service.get_recommended_speed.return_value = (None, False)
        
        decision, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        assert decision.decision_type == "FALLBACK"
        assert decision.final_speed == 80


class TestSpeedLimitSystemOverride:
    """Test suite for human override functionality."""
    
    @pytest.fixture
    def system(self):
        """Fixture providing a SpeedLimitSystem instance."""
        return SpeedLimitSystem()
    
    def test_override_valid_speed(self, system):
        """Override: Valid speed (30-130) → OVERRIDE decision."""
        original_decision = SpeedDecision(100, "NN_ONLY")
        override_speed = 70
        
        new_decision = system.apply_override(original_decision, override_speed)
        
        assert new_decision.final_speed == 70
        assert new_decision.decision_type == "OVERRIDE"
    
    def test_override_at_min_boundary(self, system):
        """Boundary: Override at minimum (30) → valid."""
        original_decision = SpeedDecision(100, "NN_ONLY")
        new_decision = system.apply_override(original_decision, 30)
        assert new_decision.final_speed == 30
        assert new_decision.decision_type == "OVERRIDE"
    
    def test_override_at_max_boundary(self, system):
        """Boundary: Override at maximum (130) → valid."""
        original_decision = SpeedDecision(100, "NN_ONLY")
        new_decision = system.apply_override(original_decision, 130)
        assert new_decision.final_speed == 130
        assert new_decision.decision_type == "OVERRIDE"
    
    def test_override_below_min_rejected(self, system):
        """Error scenario: Override below minimum (29) → rejected."""
        original_decision = SpeedDecision(100, "NN_ONLY")
        new_decision = system.apply_override(original_decision, 29)
        assert new_decision.final_speed == 100  # Original unchanged
        assert new_decision.decision_type == "NN_ONLY"  # Original type
    
    def test_override_above_max_rejected(self, system):
        """Error scenario: Override above maximum (131) → rejected."""
        original_decision = SpeedDecision(100, "NN_ONLY")
        new_decision = system.apply_override(original_decision, 131)
        assert new_decision.final_speed == 100  # Original unchanged
        assert new_decision.decision_type == "NN_ONLY"  # Original type


class TestSpeedLimitSystemThrottling:
    """Test suite for throttling/caching behavior (REQ-9)."""
    
    @pytest.fixture
    def system(self):
        """Fixture providing a SpeedLimitSystem instance."""
        return SpeedLimitSystem()
    
    @pytest.fixture
    def weather_router(self):
        return WeatherRouter()
    
    @pytest.fixture
    def air_router(self):
        return AirQualityRouter()
    
    @pytest.fixture
    def nn_model(self):
        model = Mock(spec=NeuralNetworkModel)
        model.predict_safe_speed.return_value = (None, False)
        return model
    
    @pytest.fixture
    def llm_service(self):
        service = Mock(spec=LLMService)
        service.get_recommended_speed.return_value = (None, False)
        return service
    
    @pytest.fixture
    def sensor_data(self):
        return SensorData(illuminance=1000, water_level=0, temperature=15, aqi=50)
    
    def test_throttling_returns_cached_result(self, system, weather_router, air_router, nn_model, llm_service, sensor_data):
        """Throttling: Returns cached result within 30 seconds."""
        # First call
        decision1, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Second call immediately after (should use cache)
        decision2, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Should be same result
        assert decision1.final_speed == decision2.final_speed
        assert decision1.decision_type == decision2.decision_type
        
        # Model should only be called once
        assert nn_model.predict_safe_speed.call_count == 1
    
    @patch('speedlimit.speed_limit_system.datetime')
    def test_throttling_expires_after_30_seconds(self, mock_datetime, system, weather_router, air_router, nn_model, llm_service, sensor_data):
        """Throttling: Cache expires after 30 seconds."""
        # First call at time 0
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 0)
        decision1, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Second call after 31 seconds (should recompute)
        mock_datetime.now.return_value = datetime(2024, 1, 1, 12, 0, 31)
        decision2, _, _ = system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Model should be called twice
        assert nn_model.predict_safe_speed.call_count == 2
    
    def test_throttling_invalidated_by_sensor_change(self, system, weather_router, air_router, nn_model, llm_service):
        """Throttling: Cache invalidated when sensor data changes."""
        sensor_data1 = SensorData(illuminance=1000, water_level=0, temperature=15, aqi=50)
        decision1, _, _ = system.compute_speed(sensor_data1, weather_router, air_router, nn_model, llm_service)
        
        # Different sensor data (should recompute)
        sensor_data2 = SensorData(illuminance=500, water_level=0, temperature=15, aqi=50)
        decision2, _, _ = system.compute_speed(sensor_data2, weather_router, air_router, nn_model, llm_service)
        
        # Model should be called twice
        assert nn_model.predict_safe_speed.call_count == 2
    
    def test_clear_cache(self, system, weather_router, air_router, nn_model, llm_service, sensor_data):
        """Throttling: clear_cache() invalidates cache."""
        # First call
        system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Clear cache
        system.clear_cache()
        
        # Second call should recompute
        system.compute_speed(sensor_data, weather_router, air_router, nn_model, llm_service)
        
        # Model should be called twice
        assert nn_model.predict_safe_speed.call_count == 2

