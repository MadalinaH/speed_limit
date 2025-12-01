"""
LLM service module for the SpeedLimit System.

This module contains the LLMService class which provides speed recommendations
based on air quality index (AQI) values. Supports two modes:
- "mock": Deterministic rule-based mapping (offline, always available)
- "grok": Real LLM integration via Groq API (requires API key)
"""

import os
import re
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass


class LLMService:
    """
    Service for obtaining LLM-based speed recommendations from air quality data.
    
    Supports two operation modes:
    - "mock": Uses deterministic AQI-to-speed mappings (default, offline)
    - "grok": Calls Grok LLM via Groq API for recommendations (requires GROQ_API_KEY)
    
    Mode is selected via SPEEDLIMIT_LLM_MODE environment variable or constructor parameter.
    If mode is "grok" but API key is missing or call fails, falls back to mock behavior.
    """
    
    def __init__(self, mode: Optional[str] = None):
        """
        Initialize LLMService with specified mode or from environment variable.
        
        Args:
            mode: Optional mode override ("mock" or "grok"). If None, reads from
                  SPEEDLIMIT_LLM_MODE environment variable. Defaults to "mock" if
                  not specified or invalid.
        """
        # Determine mode: explicit parameter > environment variable > default
        env_mode = os.getenv("SPEEDLIMIT_LLM_MODE", "").lower()
        self.mode = mode.lower() if mode else (env_mode if env_mode in ["mock", "grok"] else "mock")
        
        # Initialize Grok client if in grok mode (will be None if unavailable)
        self._grok_client = None
        if self.mode == "grok":
            self._grok_client = self._init_grok_client()
            if self._grok_client is None:
                # Fall back to mock if Grok initialization fails
                self.mode = "mock"
                print("LLMService: Grok mode requested but unavailable, falling back to mock mode")
    
    def _init_grok_client(self):
        """
        Initialize Groq client for Grok API calls.
        
        Returns:
            Groq client instance if successful, None otherwise
        """
        try:
            from groq import Groq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print("LLMService: GROQ_API_KEY not found, Grok mode unavailable")
                return None
            
            return Groq(api_key=api_key)
        except ImportError:
            print("LLMService: groq package not available, Grok mode unavailable")
            return None
        except Exception as e:
            print(f"LLMService: Failed to initialize Grok client: {e}")
            return None
    
    def get_recommended_speed(self, aqi: int) -> tuple[Optional[int], bool]:
        """
        Gets a recommended speed based on air quality index.
        
        Delegates to mock or grok implementation based on configured mode.
        In grok mode, falls back to mock if LLM call fails.
        
        Args:
            aqi: Air Quality Index value (higher values indicate worse air quality)
        
        Returns:
            A tuple containing:
            - Optional[int]: Recommended speed in km/h (30-130), or None if unable to determine
            - bool: True if the recommendation is valid, False otherwise
        """
        if self.mode == "grok":
            result = self._recommend_speed_grok(aqi)
            # If Grok call failed, fall back to mock
            if result[1] is False:
                return self._recommend_speed_mock(aqi)
            return result
        else:
            return self._recommend_speed_mock(aqi)
    
    def _recommend_speed_mock(self, aqi: int) -> tuple[Optional[int], bool]:
        """
        Mock implementation: deterministic AQI-to-speed mapping.
        
        This is the original rule-based implementation that maps AQI ranges
        to recommended safe speeds. Higher AQI values (worse air quality)
        result in lower recommended speeds to reduce emissions and exposure.
        
        Args:
            aqi: Air Quality Index value
        
        Returns:
            Tuple of (recommended_speed, valid) where speed is in [30, 130] km/h
        """
        # AQI > 300: Very unhealthy - recommend 30 km/h (minimum safe speed)
        if aqi > 300:
            return (30, True)
        # AQI 200-300: Unhealthy - recommend 40 km/h
        elif aqi > 200:
            return (40, True)
        # AQI 150-200: Unhealthy for sensitive groups - recommend 50 km/h
        elif aqi > 150:
            return (50, True)
        # AQI 100-150: Unhealthy for sensitive groups - recommend 60 km/h
        elif aqi > 100:
            return (60, True)
        # AQI <= 100: Should not call this method (handled by router)
        # But if called, return invalid
        else:
            return (None, False)
    
    def _recommend_speed_grok(self, aqi: int) -> tuple[Optional[int], bool]:
        """
        Grok LLM implementation: calls Grok API for speed recommendation.
        
        Sends AQI value to Grok LLM with explicit instructions to return only
        a number between 30-130 km/h. Validates and clamps the response.
        
        Args:
            aqi: Air Quality Index value
        
        Returns:
            Tuple of (recommended_speed, valid) where speed is in [30, 130] km/h,
            or (None, False) if LLM call fails or returns invalid value
        """
        if self._grok_client is None:
            return (None, False)
        
        try:
            # Build explicit prompt for the LLM
            prompt = (
                f"Given an Air Quality Index (AQI) of {aqi}, recommend a safe highway speed in km/h. "
                f"Higher AQI means worse air quality, so lower speeds reduce emissions. "
                f"The legal speed range is 30-130 km/h. "
                f"Return ONLY a number between 30 and 130, with no explanation or other text."
            )
            
            # Call Grok API
            chat_completion = self._grok_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-70b-versatile",  # Using a Grok-compatible model
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=10  # Only need a number
            )
            
            # Extract response text
            response_text = chat_completion.choices[0].message.content.strip()
            
            # Parse numeric value from response
            speed = self._parse_speed_from_response(response_text)
            
            # Validate and clamp to legal range
            if speed is not None:
                speed = max(30, min(130, speed))  # Clamp to [30, 130]
                return (speed, True)
            else:
                print(f"LLMService: Could not parse speed from Grok response: {response_text}")
                return (None, False)
                
        except Exception as e:
            print(f"LLMService: Grok API call failed: {e}")
            return (None, False)
    
    def _parse_speed_from_response(self, response_text: str) -> Optional[int]:
        """
        Extract the first numeric value from LLM response text.
        
        Args:
            response_text: Raw text response from LLM
        
        Returns:
            First integer found in the text, or None if no valid number found
        """
        # Remove common prefixes/suffixes and extract first number
        # Handle cases like "90", "90 km/h", "The speed is 90", etc.
        response_text = response_text.strip()
        
        # Try to find first integer in the text
        numbers = re.findall(r'\d+', response_text)
        if numbers:
            try:
                speed = int(numbers[0])
                return speed
            except (ValueError, IndexError):
                pass
        
        return None

