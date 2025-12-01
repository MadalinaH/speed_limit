"""
Tests for LLMService component.

Tests cover:
- Equivalence classes: different AQI ranges in mock mode
- Boundary value analysis: AQI thresholds (100, 150, 200, 300)
- Error scenarios: missing API key, API failures, invalid responses
- Full decision path coverage: mock mode, grok mode, fallback behavior
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from speedlimit.llm_service import LLMService


class TestLLMServiceMockMode:
    """Test suite for LLMService in mock mode (deterministic behavior)."""
    
    @pytest.fixture
    def llm_service_mock(self):
        """Fixture providing LLMService in mock mode."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "mock"}, clear=False):
            return LLMService()
    
    # ==================== Equivalence Classes ====================
    
    def test_aqi_very_unhealthy_mock(self, llm_service_mock):
        """Equivalence class: Very unhealthy AQI (> 300) → 30 km/h."""
        speed, valid = llm_service_mock.get_recommended_speed(350)
        assert speed == 30
        assert valid is True
    
    def test_aqi_unhealthy_high_mock(self, llm_service_mock):
        """Equivalence class: Unhealthy AQI (200-300) → 40 km/h."""
        speed, valid = llm_service_mock.get_recommended_speed(250)
        assert speed == 40
        assert valid is True
    
    def test_aqi_unhealthy_moderate_mock(self, llm_service_mock):
        """Equivalence class: Unhealthy for sensitive (150-200) → 50 km/h."""
        speed, valid = llm_service_mock.get_recommended_speed(175)
        assert speed == 50
        assert valid is True
    
    def test_aqi_unhealthy_sensitive_mock(self, llm_service_mock):
        """Equivalence class: Unhealthy for sensitive (100-150) → 60 km/h."""
        speed, valid = llm_service_mock.get_recommended_speed(120)
        assert speed == 60
        assert valid is True
    
    def test_aqi_good_invalid_mock(self, llm_service_mock):
        """Equivalence class: Good AQI (<= 100) → invalid (shouldn't call LLM)."""
        speed, valid = llm_service_mock.get_recommended_speed(80)
        assert speed is None
        assert valid is False
    
    # ==================== Boundary Value Analysis ====================
    
    def test_aqi_boundary_100_mock(self, llm_service_mock):
        """Boundary: AQI exactly at 100."""
        speed, valid = llm_service_mock.get_recommended_speed(100)
        assert speed is None
        assert valid is False
    
    def test_aqi_boundary_101_mock(self, llm_service_mock):
        """Boundary: AQI just above 100."""
        speed, valid = llm_service_mock.get_recommended_speed(101)
        assert speed == 60
        assert valid is True
    
    def test_aqi_boundary_150_mock(self, llm_service_mock):
        """Boundary: AQI exactly at 150."""
        # AQI 150 falls into range 100-150 (aqi > 100 and not > 150), so returns 60
        speed, valid = llm_service_mock.get_recommended_speed(150)
        assert speed == 60  # 150 is in range 100-150, not 150-200
        assert valid is True
    
    def test_aqi_boundary_151_mock(self, llm_service_mock):
        """Boundary: AQI just above 150."""
        speed, valid = llm_service_mock.get_recommended_speed(151)
        assert speed == 50
        assert valid is True
    
    def test_aqi_boundary_200_mock(self, llm_service_mock):
        """Boundary: AQI exactly at 200."""
        # AQI 200 falls into range 150-200 (aqi > 150 and not > 200), so returns 50
        speed, valid = llm_service_mock.get_recommended_speed(200)
        assert speed == 50  # 200 is in range 150-200, not 200-300
        assert valid is True
    
    def test_aqi_boundary_201_mock(self, llm_service_mock):
        """Boundary: AQI just above 200."""
        speed, valid = llm_service_mock.get_recommended_speed(201)
        assert speed == 40
        assert valid is True
    
    def test_aqi_boundary_300_mock(self, llm_service_mock):
        """Boundary: AQI exactly at 300."""
        # AQI 300 falls into range 200-300 (aqi > 200 and not > 300), so returns 40
        speed, valid = llm_service_mock.get_recommended_speed(300)
        assert speed == 40  # 300 is in range 200-300, not > 300
        assert valid is True
    
    def test_aqi_boundary_301_mock(self, llm_service_mock):
        """Boundary: AQI just above 300."""
        speed, valid = llm_service_mock.get_recommended_speed(301)
        assert speed == 30
        assert valid is True


class TestLLMServiceGrokMode:
    """Test suite for LLMService in grok mode (with mocked API calls)."""
    
    @pytest.fixture
    def llm_service_grok(self):
        """Fixture providing LLMService in grok mode with mocked client."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "grok", "GROQ_API_KEY": "test_key"}, clear=False):
            with patch('speedlimit.llm_service.Groq') as mock_grok:
                mock_client = MagicMock()
                mock_grok.return_value = mock_client
                service = LLMService()
                service._grok_client = mock_client
                return service
    
    def test_grok_successful_response_valid_speed(self, llm_service_grok):
        """Grok mode: Successful API call with valid speed response."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "90"
        llm_service_grok._grok_client.chat.completions.create.return_value = mock_response
        
        speed, valid = llm_service_grok.get_recommended_speed(150)
        assert speed == 90
        assert valid is True
        llm_service_grok._grok_client.chat.completions.create.assert_called_once()
    
    def test_grok_response_with_text_extracts_number(self, llm_service_grok):
        """Grok mode: Response with text, extracts number correctly."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The recommended speed is 75 km/h"
        llm_service_grok._grok_client.chat.completions.create.return_value = mock_response
        
        speed, valid = llm_service_grok.get_recommended_speed(150)
        assert speed == 75
        assert valid is True
    
    def test_grok_response_out_of_range_clamped(self, llm_service_grok):
        """Grok mode: Response out of range is clamped to [30, 130]."""
        # Test speed too high
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "200"
        llm_service_grok._grok_client.chat.completions.create.return_value = mock_response
        
        speed, valid = llm_service_grok.get_recommended_speed(150)
        assert speed == 130  # Clamped to max
        assert valid is True
        
        # Test speed too low
        mock_response.choices[0].message.content = "10"
        speed, valid = llm_service_grok.get_recommended_speed(150)
        assert speed == 30  # Clamped to min
        assert valid is True
    
    def test_grok_response_no_number_falls_back_to_mock(self, llm_service_grok):
        """Grok mode: Response with no number falls back to mock."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "I cannot determine the speed"
        llm_service_grok._grok_client.chat.completions.create.return_value = mock_response
        
        speed, valid = llm_service_grok.get_recommended_speed(150)
        # Should fall back to mock mode (AQI 150 → 50 km/h)
        assert speed == 50
        assert valid is True
    
    def test_grok_api_error_falls_back_to_mock(self, llm_service_grok):
        """Grok mode: API error falls back to mock."""
        llm_service_grok._grok_client.chat.completions.create.side_effect = Exception("API Error")
        
        speed, valid = llm_service_grok.get_recommended_speed(150)
        # Should fall back to mock mode
        assert speed == 50
        assert valid is True
    
    def test_grok_missing_api_key_falls_back_to_mock(self):
        """Grok mode: Missing API key falls back to mock at initialization."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "grok"}, clear=True):
            # Remove GROQ_API_KEY
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
            
            service = LLMService()
            # Should have fallen back to mock mode
            assert service.mode == "mock"
            
            # Should work in mock mode
            speed, valid = service.get_recommended_speed(150)
            assert speed == 50
            assert valid is True
    
    def test_grok_import_error_falls_back_to_mock(self):
        """Grok mode: Groq package not available falls back to mock."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "grok", "GROQ_API_KEY": "test_key"}, clear=False):
            with patch('speedlimit.llm_service.Groq', side_effect=ImportError("No module named 'groq'")):
                service = LLMService()
                assert service.mode == "mock"
                
                speed, valid = service.get_recommended_speed(150)
                assert speed == 50
                assert valid is True


class TestLLMServiceModeSelection:
    """Test suite for LLMService mode selection."""
    
    def test_default_mode_is_mock(self):
        """Mode selection: Defaults to mock when no env var set."""
        with patch.dict(os.environ, {}, clear=True):
            service = LLMService()
            assert service.mode == "mock"
    
    def test_env_var_mock_mode(self):
        """Mode selection: Environment variable sets mock mode."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "mock"}, clear=False):
            service = LLMService()
            assert service.mode == "mock"
    
    def test_env_var_grok_mode(self):
        """Mode selection: Environment variable sets grok mode."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "grok", "GROQ_API_KEY": "test_key"}, clear=False):
            with patch('speedlimit.llm_service.Groq'):
                service = LLMService()
                assert service.mode == "grok"
    
    def test_constructor_override_mock(self):
        """Mode selection: Constructor parameter overrides env var."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "grok"}, clear=False):
            service = LLMService(mode="mock")
            assert service.mode == "mock"
    
    def test_constructor_override_grok(self):
        """Mode selection: Constructor parameter overrides env var."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "mock"}, clear=False):
            with patch('speedlimit.llm_service.Groq'):
                with patch.dict(os.environ, {"GROQ_API_KEY": "test_key"}, clear=False):
                    service = LLMService(mode="grok")
                    assert service.mode == "grok"
    
    def test_invalid_env_var_defaults_to_mock(self):
        """Mode selection: Invalid env var value defaults to mock."""
        with patch.dict(os.environ, {"SPEEDLIMIT_LLM_MODE": "invalid"}, clear=False):
            service = LLMService()
            assert service.mode == "mock"

