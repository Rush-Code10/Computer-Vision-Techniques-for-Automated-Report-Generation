#!/usr/bin/env python3
"""
Simple system validation test for the standardized reporting system.
Tests core functionality with Orlando airport images.
"""

import os
import sys
import json
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_functionality():
    """Test basic system functionality."""
    print("🧪 Testing basic system functionality...")
    
    # Check required files
    required_files = [
        "orlando2010.png",
        "orlando2023.png",
        "config.yaml",
        "cli.py",
        "unified_runner.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files present")
    
    # Test imports
    try:
        from unified_runner import ChangeDetectionRunner
        from config_manager import ConfigManager
        from standardized_data_models import ChangeDetectionResult
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test configuration loading
    try:
        config_manager = ConfigManager()
        config = config_manager.load_config()
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False
    
    # Test runner creation
    try:
        runner = ChangeDetectionRunner()
        print("✅ Runner created successfully")
    except Exception as e:
        print(f"❌ Runner creation failed: {e}")
        return False
    
    return True

def test_single_implementation():
    """Test running a single implementation."""
    print("\n🧪 Testing single implementation...")
    
    try:
        from unified_runner import ChangeDetectionRunner
        
        runner = ChangeDetectionRunner()
        
        # Test basic implementation with Orlando images
        result = runner.run_implementation(
            'basic', 
            'orlando2010.png', 
            'orlando2023.png',
            min_area=100
        )
        
        if not result:
            print("❌ No result returned")
            return False
        
        if not hasattr(result, 'implementation_name'):
            print("❌ Invalid result structure")
            return False
        
        print(f"✅ Basic implementation completed in {result.processing_time:.3f}s")
        print(f"   • Change pixels: {result.total_change_pixels:,}")
        print(f"   • Change regions: {result.num_change_regions}")
        print(f"   • Average confidence: {result.average_confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single implementation test failed: {e}")
        return False

def test_cli_help():
    """Test CLI help functionality."""
    print("\n🧪 Testing CLI help...")
    
    try:
        import subprocess
        
        # Test help command
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print(f"❌ CLI help failed with return code {result.returncode}")
            return False
        
        if "Change Detection System" not in result.stdout:
            print("❌ CLI help output doesn't contain expected text")
            return False
        
        print("✅ CLI help working correctly")
        return True
        
    except Exception as e:
        print(f"❌ CLI help test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 Starting system validation...")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Single Implementation", test_single_implementation),
        ("CLI Help", test_cli_help)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print("VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ System is ready for use")
    else:
        print("⚠️  Some validation tests failed")
        print("❌ Review issues before using the system")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)