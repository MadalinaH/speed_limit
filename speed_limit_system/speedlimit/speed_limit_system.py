"""
Speed limit system module for the SpeedLimit System.

This module contains the SpeedLimitSystem class, which is the core orchestrator
of the speed limit decision-making process. It coordinates sensor data validation,
weather and air quality routing, neural network and LLM model predictions, and
combines their recommendations according to defined rules to produce a final
speed decision with explanations and logging.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .sensor_data import SensorData
from .weather_router import WeatherRouter
from .air_quality_router import AirQualityRouter
from .nn_model import NeuralNetworkModel
from .llm_service import LLMService
from .model_recommendations import ModelRecommendations
from .speed_decision import SpeedDecision
from .explanation import Explanation
from .log_entry import LogEntry


class SpeedLimitSystem:
    """
    Core orchestrator for the speed limit decision-making system.
    
    Coordinates all components of the speed limit system: validates sensor data,
    routes decisions through weather and air quality analyzers, calls neural
    network and LLM models directly when conditions require it, combines their
    recommendations, enforces safety constraints, and produces final decisions
    with explanations and log entries.
    
    Enforces REQ-9: updates are throttled to at most once every 30 seconds.
    Repeated calls within 30 seconds return the last computed result without
    recomputing. This ensures all callers (including Live mode UI) respect the
    update frequency limit.
    """
    
    # Legal speed range: 30-130 km/h
    MIN_SPEED = 30
    MAX_SPEED = 130
    DEFAULT_SPEED = 130  # Default highway speed when no hazards
    FALLBACK_SPEED = 80  # Conservative fallback speed on errors
    
    # Update throttling: minimum 30 seconds between computations (REQ-9)
    MIN_UPDATE_INTERVAL_SECONDS = 30
    
    # Cache version: increment this to invalidate all cached results (useful after code changes)
    CACHE_VERSION = 2
    
    # Logging configuration
    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / "system_log.log"
    
    def __init__(self):
        """Initialize the speed limit system with throttling state."""
        self._last_computation_time: Optional[datetime] = None
        self._last_result: Optional[tuple[SpeedDecision, Explanation, LogEntry]] = None
        self._last_sensor_data_hash: Optional[str] = None
        self._cache_version: int = self.CACHE_VERSION
        
        # Initialize log directory and file if needed
        self._ensure_log_file_exists()
    
    def clear_cache(self) -> None:
        """Clear the throttling cache to force next computation."""
        self._last_computation_time = None
        self._last_result = None
        self._last_sensor_data_hash = None
        self._cache_version = self.CACHE_VERSION
    
    def _ensure_log_file_exists(self) -> None:
        """Create log directory if needed."""
        self.LOG_DIR.mkdir(exist_ok=True)
        
        # Create log file with header if it doesn't exist
        if not self.LOG_FILE.exists():
            with open(self.LOG_FILE, 'w', encoding='utf-8') as f:
                f.write("# Speed Limit System Log\n")
                f.write("# Format: [TIMESTAMP] SPEED(km/h) | TYPE | HAZARDS | MODELS | REASONING | REUSED\n")
                f.write("# " + "=" * 80 + "\n\n")
    
    def _log_decision(
        self,
        decision: SpeedDecision,
        log_entry: LogEntry,
        explanation: Explanation,
        result_reused: bool = False
    ) -> None:
        """
        Log an automatic decision to the persistent log file.
        
        Args:
            decision: The speed decision to log
            log_entry: The log entry with details
            explanation: The explanation text
            result_reused: True if this result was from cache (throttled), False if newly computed
        """
        # Extract hazard flags
        weather_flag_str = log_entry.details.get("weather_reduction_needed", "False")
        poor_air_str = log_entry.details.get("poor_air_quality", "False")
        weather_hazard = weather_flag_str.lower() == "true" if isinstance(weather_flag_str, str) else bool(weather_flag_str)
        poor_air = poor_air_str.lower() == "true" if isinstance(poor_air_str, str) else bool(poor_air_str)
        
        # Extract model speeds
        nn_speed_str = log_entry.details.get("nn_speed", "None")
        llm_speed_str = log_entry.details.get("llm_speed", "None")
        nn_speed = nn_speed_str if nn_speed_str != "None" else ""
        llm_speed = llm_speed_str if llm_speed_str != "None" else ""
        
        # Create high-level reasoning summary
        reasoning_parts = []
        if decision.decision_type == "COMBINED":
            reasoning_parts.append("Both NN and LLM models contributed")
        elif decision.decision_type == "NN_ONLY":
            reasoning_parts.append("Neural network model contributed")
        elif decision.decision_type == "LLM_ONLY":
            reasoning_parts.append("LLM service contributed")
        elif decision.decision_type == "DEFAULT":
            reasoning_parts.append("Normal conditions, no model outputs needed")
        elif decision.decision_type == "FALLBACK":
            reasoning_parts.append("System fallback due to invalid inputs or model failure")
        
        if weather_hazard:
            reasoning_parts.append("Weather conditions detected")
        if poor_air:
            reasoning_parts.append("Poor air quality detected")
        
        if result_reused:
            reasoning_parts.append("(Result reused from cache due to throttling)")
        
        reasoning_summary = "; ".join(reasoning_parts) if reasoning_parts else "No specific reasoning"
        
        # Format hazards
        hazards_list = []
        if weather_hazard:
            hazards_list.append("Weather")
        if poor_air:
            hazards_list.append("Air")
        hazards_str = ", ".join(hazards_list) if hazards_list else "None"
        
        # Format model speeds
        models_str = ""
        if nn_speed and llm_speed:
            models_str = f"NN:{nn_speed}, LLM:{llm_speed}"
        elif nn_speed:
            models_str = f"NN:{nn_speed}"
        elif llm_speed:
            models_str = f"LLM:{llm_speed}"
        else:
            models_str = "None"
        
        # Format reused indicator
        reused_str = "[REUSED]" if result_reused else "[NEW]"
        
        # Write to log file in human-readable format
        try:
            with open(self.LOG_FILE, 'a', encoding='utf-8') as f:
                timestamp_str = log_entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                log_line = (
                    f"[{timestamp_str}] {decision.final_speed:3d} km/h | "
                    f"{decision.decision_type:10s} | "
                    f"Hazards: {hazards_str:15s} | "
                    f"Models: {models_str:20s} | "
                    f"{reasoning_summary} | "
                    f"{reused_str}\n"
                )
                f.write(log_line)
        except Exception as e:
            # Logging should not break the system - silently fail or could use Python logging
            pass
    
    def compute_speed(
        self,
        sensor_data: SensorData,
        weather_router: WeatherRouter,
        air_router: AirQualityRouter,
        nn_model: NeuralNetworkModel,
        llm_service: LLMService,
        enable_persistent_logging: bool = False
    ) -> tuple[SpeedDecision, Explanation, LogEntry]:
        """
        Computes the final speed limit decision based on sensor data and models.
        
        This is the main decision-making method that orchestrates the entire
        workflow: validation, condition assessment, model predictions, recommendation
        combination, safety enforcement, and result generation.
        
        Enforces REQ-9 throttling: if called within 30 seconds of the last computation,
        returns the cached result without recomputing. This ensures all callers respect
        the update frequency limit.
        
        Args:
            sensor_data: The sensor readings to process
            weather_router: Router for weather-based speed reduction decisions
            air_router: Router for air quality classification (determines if air is poor)
            nn_model: Neural network model for safe speed prediction
            llm_service: LLM service for air quality recommendations (called directly when poor air is detected)
            enable_persistent_logging: If True, log this decision to persistent log file (for Live mode)
        
        Returns:
            A tuple containing:
            - SpeedDecision: The final speed decision with type
            - Explanation: Human-readable explanation of the decision
            - LogEntry: Complete log record of the decision cycle
        """
        # REQ-9: Throttle updates to at most once every 30 seconds
        # Create a simple hash of sensor data to detect changes
        sensor_hash = f"{sensor_data.illuminance}_{sensor_data.water_level}_{sensor_data.temperature}_{sensor_data.aqi}"
        
        now = datetime.now()
        result_reused = False
        
        # Check cache version to invalidate old cached results after code changes
        if (self._cache_version != self.CACHE_VERSION):
            # Cache version mismatch: clear cache
            self._last_computation_time = None
            self._last_result = None
            self._last_sensor_data_hash = None
            self._cache_version = self.CACHE_VERSION
        
        if (self._last_computation_time is not None and 
            self._last_result is not None and
            self._last_sensor_data_hash == sensor_hash and
            (now - self._last_computation_time).total_seconds() < self.MIN_UPDATE_INTERVAL_SECONDS):
            # Return cached result only if within throttling window AND sensor data hasn't changed
            decision, explanation, log_entry = self._last_result
            
            # Log cached result if logging is enabled (REQ-9: log even throttled results)
            if enable_persistent_logging:
                self._log_decision(decision, log_entry, explanation, result_reused=True)
            
            return self._last_result
        
        # Step 1: Validate SensorData
        valid, reason = sensor_data.validate()
        if not valid:
            # Invalid sensor data: use conservative fallback speed
            final_speed = self.FALLBACK_SPEED
            decision_type = "FALLBACK"
            explanation_text = f"Input sensor values invalid: {reason}. Using fallback speed {self.FALLBACK_SPEED} km/h."
            
            # Create decision, explanation, and log entry
            decision = SpeedDecision(final_speed, decision_type)
            explanation = Explanation(explanation_text)
            log_entry = LogEntry(
                timestamp=datetime.now(),
                sensor_data=sensor_data,
                decision=decision,
                details={
                    "validation_error": reason or "unknown",
                    "reason": "Invalid sensor data"
                }
            )
            return (decision, explanation, log_entry)
        
        # Step 2: Determine conditions
        # Check if weather conditions require speed reduction
        weather_flag = weather_router.should_reduce_for_weather(sensor_data)
        # Check if air quality is poor
        poor_air = air_router.is_poor_air_quality(sensor_data)
        
        # Step 3: Model calls
        # Get neural network prediction (only if weather reduction is needed)
        nn_speed_raw, nn_valid_raw = nn_model.predict_safe_speed(sensor_data, weather_flag)
        # Get LLM-based air quality recommendation (only if poor air quality detected)
        if poor_air:
            llm_speed_raw, llm_valid_raw = llm_service.get_recommended_speed(sensor_data.aqi)
        else:
            llm_speed_raw, llm_valid_raw = (None, False)
        
        # Validate model outputs are within legal range [30, 130]
        # Model outputs outside this range are treated as invalid
        nn_speed = None
        nn_valid = False
        if nn_valid_raw and nn_speed_raw is not None:
            if self.MIN_SPEED <= nn_speed_raw <= self.MAX_SPEED:
                nn_speed = nn_speed_raw
                nn_valid = True
        
        llm_speed = None
        llm_valid = False
        if llm_valid_raw and llm_speed_raw is not None:
            if self.MIN_SPEED <= llm_speed_raw <= self.MAX_SPEED:
                llm_speed = llm_speed_raw
                llm_valid = True
        
        # Step 4: Build ModelRecommendations instance
        recommendations = ModelRecommendations(
            nn_speed=nn_speed,
            nn_valid=nn_valid,
            llm_speed=llm_speed,
            llm_valid=llm_valid
        )
        
        # Step 5: Combine recommendations
        # Determine final speed and decision type based on model validity
        if recommendations.both_valid():
            # Both models valid: use the lower (more conservative) speed
            final_speed = min(nn_speed, llm_speed)
            decision_type = "COMBINED"
        elif recommendations.nn_valid:
            # Only NN valid: use NN recommendation
            final_speed = nn_speed
            decision_type = "NN_ONLY"
        elif recommendations.llm_valid:
            # Only LLM valid: use LLM recommendation
            final_speed = llm_speed
            decision_type = "LLM_ONLY"
        else:
            # Neither model valid: determine if this is normal conditions or system failure
            if not weather_flag and not poor_air:
                # No hazards detected: normal conditions, use default highway speed (130 km/h)
                final_speed = self.DEFAULT_SPEED
                decision_type = "DEFAULT"
            else:
                # Hazards detected but models failed to provide usable output: system failure, use fallback (80 km/h)
                final_speed = self.FALLBACK_SPEED
                decision_type = "FALLBACK"
        
        # Step 6: Safety enforcement
        # Ensure final speed is within legal range [30, 130]
        if final_speed < self.MIN_SPEED or final_speed > self.MAX_SPEED:
            final_speed = self.FALLBACK_SPEED
            decision_type = "FALLBACK"
        
        # Step 7: Create result objects
        decision = SpeedDecision(final_speed, decision_type)
        
        # Build human-readable explanation
        explanation_parts = []
        if decision_type == "COMBINED":
            explanation_parts.append(f"Both models recommended speeds: NN={nn_speed} km/h, LLM={llm_speed} km/h. Using lower speed {final_speed} km/h.")
        elif decision_type == "NN_ONLY":
            explanation_parts.append(f"Neural network recommended {final_speed} km/h based on weather conditions.")
        elif decision_type == "LLM_ONLY":
            explanation_parts.append(f"LLM service recommended {final_speed} km/h based on air quality.")
        elif decision_type == "DEFAULT":
            explanation_parts.append(f"Normal conditions detected. Using default highway speed {self.DEFAULT_SPEED} km/h.")
        elif decision_type == "FALLBACK":
            explanation_parts.append(f"Using fallback speed {self.FALLBACK_SPEED} km/h due to invalid model outputs or safety constraints.")
        
        # Add condition context
        if weather_flag:
            explanation_parts.append("Weather conditions require speed reduction.")
        if poor_air:
            explanation_parts.append("Poor air quality detected.")
        
        explanation_text = " ".join(explanation_parts) if explanation_parts else f"Speed set to {final_speed} km/h."
        explanation = Explanation(explanation_text)
        
        # Build log entry with details
        log_entry = LogEntry(
            timestamp=datetime.now(),
            sensor_data=sensor_data,
            decision=decision,
            details={
                "weather_reduction_needed": str(weather_flag),
                "poor_air_quality": str(poor_air),
                "nn_speed": str(nn_speed) if nn_speed is not None else "None",
                "nn_valid": str(nn_valid),
                "llm_speed": str(llm_speed) if llm_speed is not None else "None",
                "llm_valid": str(llm_valid),
            }
        )
        
        # Cache result for throttling (REQ-9)
        self._last_computation_time = now
        self._last_result = (decision, explanation, log_entry)
        self._last_sensor_data_hash = sensor_hash
        
        # Log to persistent file if enabled (for Live mode)
        if enable_persistent_logging:
            self._log_decision(decision, log_entry, explanation, result_reused=False)
        
        return (decision, explanation, log_entry)
    
    def apply_override(
        self,
        decision: SpeedDecision,
        override_speed: int
    ) -> SpeedDecision:
        """
        Applies a human override to a speed decision.
        
        Allows a human operator to override the system's decision with a
        manually specified speed. The override is only applied if it falls
        within the legal operating range [30, 130] km/h. Invalid overrides
        are rejected and the original decision is returned unchanged.
        
        Args:
            decision: The original speed decision to potentially override
            override_speed: The human-specified speed to use (must be 30-130)
        
        Returns:
            A new SpeedDecision with type "OVERRIDE" if override is valid,
            or the original decision if override is invalid
        """
        # Validate override speed is within legal range [30, 130]
        if self.MIN_SPEED <= override_speed <= self.MAX_SPEED:
            # Valid override: create new decision with OVERRIDE type
            return SpeedDecision(override_speed, "OVERRIDE")
        else:
            # Invalid override: return original decision unchanged
            return decision

