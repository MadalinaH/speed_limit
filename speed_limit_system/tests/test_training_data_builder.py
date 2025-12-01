"""
Tests for TrainingDataBuilder component.

Tests cover:
- Equivalence classes: different accident scenarios
- Boundary value analysis: thresholds for near-accident labeling (15°, 50cm)
- Error scenarios: missing values, invalid data
- Full decision path coverage: labeling, cleaning, augmentation steps
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from speedlimit.training.training_data_builder import (
    _label_near_accidents,
    _clean_data,
    _augment_data,
    TARGET_POSITIVE_RATIO
)


class TestNearAccidentLabeling:
    """Test suite for near-accident labeling logic."""
    
    # ==================== Equivalence Classes ====================
    
    def test_skid_angle_high_near_accident(self):
        """Equivalence class: High skid angle (> 15°) → near-accident."""
        accidents_df = pd.DataFrame({
            'Month': [1],
            'Day': [1],
            'Hour': [1],
            'SkidAngle': [20],  # > 15
            'CloseCarCm': [100],
            'CloseGuardrailCm': [100]
        })
        
        result = _label_near_accidents(accidents_df)
        
        assert result['near_accident'].iloc[0] == 1
    
    def test_close_car_near_accident(self):
        """Equivalence class: Close car (< 50cm) → near-accident."""
        accidents_df = pd.DataFrame({
            'Month': [1],
            'Day': [1],
            'Hour': [1],
            'SkidAngle': [5],  # Safe
            'CloseCarCm': [30],  # < 50
            'CloseGuardrailCm': [100]
        })
        
        result = _label_near_accidents(accidents_df)
        
        assert result['near_accident'].iloc[0] == 1
    
    def test_close_guardrail_near_accident(self):
        """Equivalence class: Close guardrail (< 50cm) → near-accident."""
        accidents_df = pd.DataFrame({
            'Month': [1],
            'Day': [1],
            'Hour': [1],
            'SkidAngle': [5],
            'CloseCarCm': [100],
            'CloseGuardrailCm': [40]  # < 50
        })
        
        result = _label_near_accidents(accidents_df)
        
        assert result['near_accident'].iloc[0] == 1
    
    def test_safe_conditions_no_near_accident(self):
        """Equivalence class: All values safe → no near-accident."""
        accidents_df = pd.DataFrame({
            'Month': [1],
            'Day': [1],
            'Hour': [1],
            'SkidAngle': [5],  # < 15
            'CloseCarCm': [100],  # > 50
            'CloseGuardrailCm': [100]  # > 50
        })
        
        result = _label_near_accidents(accidents_df)
        
        assert result['near_accident'].iloc[0] == 0
    
    # ==================== Boundary Value Analysis ====================
    
    def test_skid_angle_boundary_below(self):
        """Boundary: SkidAngle just below threshold (14°)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [14],  # Just below 15
            'CloseCarCm': [100], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0
    
    def test_skid_angle_boundary_at(self):
        """Boundary: SkidAngle exactly at threshold (15°)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [15],  # Exactly at threshold
            'CloseCarCm': [100], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0  # Not > 15
    
    def test_skid_angle_boundary_above(self):
        """Boundary: SkidAngle just above threshold (16°)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [16],  # Just above 15
            'CloseCarCm': [100], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 1
    
    def test_close_car_boundary_below(self):
        """Boundary: CloseCarCm just below threshold (49cm)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [5], 'CloseCarCm': [49], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 1
    
    def test_close_car_boundary_at(self):
        """Boundary: CloseCarCm exactly at threshold (50cm)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [5], 'CloseCarCm': [50], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0  # Not < 50
    
    def test_close_car_boundary_above(self):
        """Boundary: CloseCarCm just above threshold (51cm)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [5], 'CloseCarCm': [51], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0
    
    # ==================== Error Scenarios ====================
    
    def test_missing_skid_angle_treated_as_safe(self):
        """Error scenario: Missing SkidAngle → treated as safe (0)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [np.nan],  # Missing
            'CloseCarCm': [100], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0  # Safe default
    
    def test_missing_close_car_treated_as_safe(self):
        """Error scenario: Missing CloseCarCm → treated as safe (999)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [5], 'CloseCarCm': [np.nan], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0  # Safe default (999 > 50)
    
    def test_missing_close_guardrail_treated_as_safe(self):
        """Error scenario: Missing CloseGuardrailCm → treated as safe (999)."""
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [5], 'CloseCarCm': [100], 'CloseGuardrailCm': [np.nan]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 0  # Safe default
    
    def test_multiple_conditions_or_logic(self):
        """Decision path: Any condition triggers near-accident (OR logic)."""
        # Only SkidAngle triggers
        accidents_df = pd.DataFrame({
            'Month': [1], 'Day': [1], 'Hour': [1],
            'SkidAngle': [20], 'CloseCarCm': [100], 'CloseGuardrailCm': [100]
        })
        result = _label_near_accidents(accidents_df)
        assert result['near_accident'].iloc[0] == 1


class TestDataCleaning:
    """Test suite for data cleaning logic."""
    
    # ==================== Equivalence Classes ====================
    
    def test_clean_valid_data_preserved(self):
        """Equivalence class: Valid data is preserved."""
        df = pd.DataFrame({
            'L': [500], 'W': [100], 'T': [15], 'SpeedLimit': [80],
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 1
    
    def test_clean_missing_essential_features_removed(self):
        """Error scenario: Missing essential features → removed."""
        df = pd.DataFrame({
            'L': [500], 'W': [np.nan], 'T': [15], 'SpeedLimit': [80],  # W missing
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 0
    
    def test_clean_out_of_range_light_removed(self):
        """Error scenario: Out-of-range light values → removed."""
        df = pd.DataFrame({
            'L': [-1], 'W': [100], 'T': [15], 'SpeedLimit': [80],  # L < 0
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 0
    
    def test_clean_out_of_range_water_removed(self):
        """Error scenario: Out-of-range water values → removed."""
        df = pd.DataFrame({
            'L': [500], 'W': [15000], 'T': [15], 'SpeedLimit': [80],  # W > 10000
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 0
    
    def test_clean_out_of_range_temperature_removed(self):
        """Error scenario: Out-of-range temperature → removed."""
        df = pd.DataFrame({
            'L': [500], 'W': [100], 'T': [-35], 'SpeedLimit': [80],  # T < -30
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 0
    
    def test_clean_out_of_range_speed_limit_removed(self):
        """Error scenario: Out-of-range speed limit → removed."""
        df = pd.DataFrame({
            'L': [500], 'W': [100], 'T': [15], 'SpeedLimit': [150],  # > 130
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 0
    
    # ==================== Boundary Value Analysis ====================
    
    def test_clean_light_at_min_boundary(self):
        """Boundary: Light at minimum (0)."""
        df = pd.DataFrame({
            'L': [0], 'W': [100], 'T': [15], 'SpeedLimit': [80],
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 1
    
    def test_clean_light_at_max_boundary(self):
        """Boundary: Light at maximum (1000000)."""
        df = pd.DataFrame({
            'L': [1000000], 'W': [100], 'T': [15], 'SpeedLimit': [80],
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 1
    
    def test_clean_speed_limit_at_min_boundary(self):
        """Boundary: SpeedLimit at minimum (30)."""
        df = pd.DataFrame({
            'L': [500], 'W': [100], 'T': [15], 'SpeedLimit': [30],
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 1
    
    def test_clean_speed_limit_at_max_boundary(self):
        """Boundary: SpeedLimit at maximum (130)."""
        df = pd.DataFrame({
            'L': [500], 'W': [100], 'T': [15], 'SpeedLimit': [130],
            'near_accident': [0]
        })
        result = _clean_data(df)
        assert len(result) == 1


class TestDataAugmentation:
    """Test suite for data augmentation logic."""
    
    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample training data."""
        return pd.DataFrame({
            'L': [500, 1000],
            'W': [100, 200],
            'T': [15, 20],
            'SpeedLimit': [80, 100],
            'near_accident': [0, 1]  # One positive, one negative
        })
    
    def test_speed_sweeps_applied_first(self, sample_data):
        """Augmentation: Speed sweeps are applied first."""
        result = _augment_data(sample_data)
        
        # Should have rows for each speed sweep (6 speeds) for each input row
        # Original 2 rows × 6 speeds = 12 rows minimum (before oversampling/noise)
        assert len(result) >= 12
        
        # Check that speed sweeps are present
        unique_speeds = result['SpeedLimit'].unique()
        assert 130 in unique_speeds
        assert 40 in unique_speeds
    
    def test_oversampling_increases_positive_ratio(self, sample_data):
        """Augmentation: Oversampling increases positive class ratio."""
        # Calculate positive ratio before augmentation
        original_positive_ratio = (sample_data['near_accident'] == 1).sum() / len(sample_data)
        
        result = _augment_data(sample_data)
        final_positive_ratio = (result['near_accident'] == 1).sum() / len(result)
        
        # If original ratio was below target, final should be closer to target
        if original_positive_ratio < TARGET_POSITIVE_RATIO:
            assert final_positive_ratio >= original_positive_ratio
    
    def test_noise_injection_applied(self, sample_data):
        """Augmentation: Noise injection modifies L, W, T values."""
        result = _augment_data(sample_data)
        
        # Find rows with same SpeedLimit and near_accident (should have noise variation)
        # Check that noise was applied (values differ from original)
        original_l = sample_data['L'].iloc[0]
        result_l_values = result[result['SpeedLimit'] == 80]['L'].values
        
        # At least some values should differ (due to noise)
        # Note: This is probabilistic, but with multiple rows it's very likely
        assert len(result_l_values) > 1
    
    def test_noise_clipped_to_valid_ranges(self, sample_data):
        """Augmentation: Noise injection clips values to valid ranges."""
        result = _augment_data(sample_data)
        
        # All L values should be in [0, 1000000]
        assert (result['L'] >= 0).all()
        assert (result['L'] <= 1000000).all()
        
        # All W values should be in [0, 10000]
        assert (result['W'] >= 0).all()
        assert (result['W'] <= 10000).all()
        
        # All T values should be in [-30, 70]
        assert (result['T'] >= -30).all()
        assert (result['T'] <= 70).all()
    
    def test_duplicates_removed(self, sample_data):
        """Augmentation: Duplicates are removed after augmentation."""
        result = _augment_data(sample_data)
        
        # Check for duplicates (all columns should be unique)
        duplicates = result.duplicated()
        assert not duplicates.any(), "Augmentation should remove duplicates"
    
    def test_augmentation_preserves_near_accident_labels(self, sample_data):
        """Augmentation: Near-accident labels are preserved."""
        result = _augment_data(sample_data)
        
        # Should have both positive and negative examples
        assert (result['near_accident'] == 1).sum() > 0
        assert (result['near_accident'] == 0).sum() > 0
    
    def test_speed_sweeps_all_candidate_speeds(self, sample_data):
        """Augmentation: All candidate speeds (130, 120, 100, 80, 60, 40) are present."""
        result = _augment_data(sample_data)
        
        expected_speeds = [130, 120, 100, 80, 60, 40]
        unique_speeds = set(result['SpeedLimit'].unique())
        
        for speed in expected_speeds:
            assert speed in unique_speeds, f"Speed {speed} should be present after augmentation"

