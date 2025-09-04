"""Test suite for CNNParted."""

import os
import sys

import pytest

# Add the cnnparted directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cnnparted"))


def test_import_cnnparted():
    """Test that cnnparted can be imported."""
    try:
        import cnnparted

        assert True
    except ImportError:
        pytest.fail("Could not import cnnparted module")


def test_import_framework():
    """Test that framework modules can be imported."""
    try:
        from framework import ConfigHelper

        assert True
    except ImportError:
        pytest.fail("Could not import framework.ConfigHelper")


def test_config_helper():
    """Test basic ConfigHelper functionality."""
    # Create a simple test config file
    test_config = """
general:
  work_dir: test_output

pipeline:
  stages: []
"""

    with open("test_config.yaml", "w") as f:
        f.write(test_config)

    try:
        from framework import ConfigHelper

        config_helper = ConfigHelper("test_config.yaml")
        config = config_helper.get_config()

        assert "general" in config
        assert "pipeline" in config
        assert config["general"]["work_dir"] == "test_output"

    finally:
        # Clean up
        if os.path.exists("test_config.yaml"):
            os.remove("test_config.yaml")


def test_workloads_import():
    """Test that workload modules can be imported."""
    try:
        from workloads import alexnet

        # Check for standard workload attributes
        assert hasattr(alexnet, "model"), "Missing 'model' attribute"
        assert hasattr(
            alexnet, "accuracy_function"
        ), "Missing 'accuracy_function' attribute"
    except ImportError:
        pytest.fail("Could not import workloads.alexnet")


def test_specialized_interfaces():
    """Test that specialized interfaces can be imported."""
    # Test MNSIM Interface with correct case
    try:
        from cnnparted.framework.node.mnsim_interface import MNSIMInterface

        assert MNSIMInterface is not None, "MNSIMInterface class should be available"
    except ImportError:
        pytest.fail("Could not import MNSIMInterface")

    # Test Timeloop Interface
    try:
        from cnnparted.framework.node.timeloop import Timeloop

        assert Timeloop is not None, "Timeloop class should be available"
    except ImportError:
        pytest.fail("Could not import Timeloop")

    # Test Zigzag Interface
    try:
        from cnnparted.framework.node.zigzag import Zigzag

        assert Zigzag is not None, "Zigzag class should be available"
    except ImportError:
        pytest.fail("Could not import Zigzag")


if __name__ == "__main__":
    pytest.main([__file__])
