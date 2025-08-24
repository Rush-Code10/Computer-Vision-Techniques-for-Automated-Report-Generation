#!/usr/bin/env python3
"""
Test script for configuration management and CLI functionality.
Tests the configuration loading, validation, and CLI argument processing.
"""

import os
import tempfile
import yaml
from config_manager import ConfigManager, SystemConfig
from logging_utils import setup_logging, ChangeDetectionLogger
import argparse


def test_config_loading():
    """Test configuration file loading."""
    print("Testing configuration loading...")
    
    # Test with default config file
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    assert isinstance(config, SystemConfig)
    assert config.implementations == ['basic', 'advanced', 'deep_learning']
    assert config.min_area_threshold == 100
    assert config.log_level == "INFO"
    
    print("✅ Configuration loading test passed")


def test_config_validation():
    """Test configuration validation."""
    print("Testing configuration validation...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Test with missing required fields
    config.image1_path = ""
    config.image2_path = ""
    errors = config_manager.validate_config()
    assert len(errors) >= 2  # Should have errors for missing image paths
    
    # Test with valid configuration
    config.image1_path = "test1.jpg"
    config.image2_path = "test2.jpg"
    
    # Create dummy files for validation
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f1:
        config.image1_path = f1.name
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f2:
        config.image2_path = f2.name
    
    try:
        errors = config_manager.validate_config()
        assert len(errors) == 0  # Should have no errors now
        print("✅ Configuration validation test passed")
    finally:
        # Clean up temp files
        os.unlink(config.image1_path)
        os.unlink(config.image2_path)


def test_config_merging():
    """Test merging configuration with command-line arguments."""
    print("Testing configuration merging...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Create mock command-line arguments
    args = argparse.Namespace()
    args.img1 = "new_image1.jpg"
    args.img2 = "new_image2.jpg"
    args.implementation = "basic"
    args.output_dir = "custom_output"
    args.min_area = 200
    args.verbose = True
    
    # Merge with args
    merged_config = config_manager.merge_with_args(args)
    
    assert merged_config.image1_path == "new_image1.jpg"
    assert merged_config.image2_path == "new_image2.jpg"
    assert merged_config.implementations == ["basic"]
    assert merged_config.base_directory == "custom_output"
    assert merged_config.min_area_threshold == 200
    assert merged_config.log_level == "DEBUG"  # verbose flag
    
    print("✅ Configuration merging test passed")


def test_logging_setup():
    """Test logging system setup."""
    print("Testing logging setup...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Test logging setup
    logger, progress_tracker, performance_monitor = setup_logging(config)
    
    assert isinstance(logger, ChangeDetectionLogger)
    assert progress_tracker is not None
    assert performance_monitor is not None
    
    # Test basic logging
    logger.info("Test log message")
    logger.debug("Test debug message")
    logger.warning("Test warning message")
    
    # Test performance monitoring
    performance_monitor.start_monitoring()
    performance_monitor.checkpoint("test_checkpoint")
    summary = performance_monitor.get_summary()
    assert 'total_time' in summary
    assert 'test_checkpoint' in summary['checkpoint_details']
    
    print("✅ Logging setup test passed")


def test_config_save_load():
    """Test saving and loading configuration."""
    print("Testing configuration save/load...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Modify some settings
    config.min_area_threshold = 150
    config.log_level = "DEBUG"
    config.implementations = ["advanced", "deep_learning"]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        temp_config_path = f.name
    
    try:
        config_manager.save_config(temp_config_path)
        
        # Load from saved file
        new_config_manager = ConfigManager(temp_config_path)
        loaded_config = new_config_manager.load_config()
        
        assert loaded_config.min_area_threshold == 150
        assert loaded_config.log_level == "DEBUG"
        assert loaded_config.implementations == ["advanced", "deep_learning"]
        
        print("✅ Configuration save/load test passed")
    finally:
        os.unlink(temp_config_path)


def test_implementation_params():
    """Test implementation-specific parameter handling."""
    print("Testing implementation parameters...")
    
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Test getting implementation parameters
    basic_params = config_manager.get_implementation_params('basic')
    assert isinstance(basic_params, dict)
    
    advanced_params = config_manager.get_implementation_params('advanced')
    assert isinstance(advanced_params, dict)
    
    deep_learning_params = config_manager.get_implementation_params('deep_learning')
    assert isinstance(deep_learning_params, dict)
    
    # Test non-existent implementation
    empty_params = config_manager.get_implementation_params('nonexistent')
    assert empty_params == {}
    
    print("✅ Implementation parameters test passed")


def run_all_tests():
    """Run all configuration and CLI tests."""
    print("=" * 50)
    print("CONFIGURATION AND CLI SYSTEM TESTS")
    print("=" * 50)
    
    try:
        test_config_loading()
        test_config_validation()
        test_config_merging()
        test_logging_setup()
        test_config_save_load()
        test_implementation_params()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Configuration and CLI system is working correctly.")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)