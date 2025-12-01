"""
Pytest configuration for SpeedLimit System tests.

Registers custom markers and provides shared fixtures.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )

