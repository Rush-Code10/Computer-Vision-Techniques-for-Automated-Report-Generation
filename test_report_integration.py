"""
Test script to verify report generation integration with the unified runner.
Tests the complete workflow including report generation.
"""

import os
import subprocess
import sys
from pathlib import Path


def test_report_generation_integration():
    """Test report generation through the unified runner."""
    print("Testing report generation integration...")
    
    # Check if required images exist
    img1 = "orlando2010.png"
    img2 = "orlando2023.png"
    
    if not os.path.exists(img1) or not os.path.exists(img2):
        print(f"❌ Required images not found: {img1}, {img2}")
        return False
    
    # Run unified runner with report generation
    cmd = [
        sys.executable, "unified_runner.py",
        img1, img2,
        "--implementation", "all",
        "--generate-reports",
        "--reports-dir", "test_reports",
        "--save",
        "--evaluate"
    ]
    
    try:
        print("Running unified runner with report generation...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Unified runner completed successfully")
            print("Output:", result.stdout[-500:])  # Show last 500 chars
            
            # Check if reports were generated
            reports_dir = Path("test_reports")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.pdf"))
                if report_files:
                    print(f"✅ Found {len(report_files)} PDF report(s):")
                    for report in report_files:
                        print(f"  • {report.name}")
                    return True
                else:
                    print("❌ No PDF reports found in test_reports directory")
                    return False
            else:
                print("❌ Reports directory not created")
                return False
        else:
            print("❌ Unified runner failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out after 120 seconds")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False


def test_individual_method_report():
    """Test report generation for a single method."""
    print("Testing individual method report generation...")
    
    # Check if required images exist
    img1 = "orlando2010.png"
    img2 = "orlando2023.png"
    
    if not os.path.exists(img1) or not os.path.exists(img2):
        print(f"❌ Required images not found: {img1}, {img2}")
        return False
    
    # Run unified runner with single implementation and report generation
    cmd = [
        sys.executable, "unified_runner.py",
        img1, img2,
        "--implementation", "basic",
        "--generate-reports",
        "--reports-dir", "test_reports_single",
        "--save"
    ]
    
    try:
        print("Running unified runner for single method...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Single method runner completed successfully")
            
            # Check if reports were generated
            reports_dir = Path("test_reports_single")
            if reports_dir.exists():
                report_files = list(reports_dir.glob("*.pdf"))
                if report_files:
                    print(f"✅ Found {len(report_files)} PDF report(s) for single method")
                    return True
                else:
                    print("❌ No PDF reports found for single method")
                    return False
            else:
                print("❌ Single method reports directory not created")
                return False
        else:
            print("❌ Single method runner failed")
            print("Error:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Single method test timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"❌ Single method test failed with exception: {e}")
        return False


def cleanup_test_files():
    """Clean up test files and directories."""
    import shutil
    
    test_dirs = ["test_reports", "test_reports_single"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            try:
                shutil.rmtree(test_dir)
                print(f"Cleaned up {test_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {test_dir}: {e}")


def main():
    """Run integration tests for report generation."""
    print("=" * 60)
    print("REPORT GENERATION INTEGRATION TESTS")
    print("=" * 60)
    
    # Clean up any existing test files
    cleanup_test_files()
    
    tests = [
        test_individual_method_report,
        test_report_generation_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            print()
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{total} integration tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All integration tests passed!")
        print("Report generation is successfully integrated with the unified runner!")
    else:
        print(f"⚠️  {total - passed} integration test(s) failed")
    
    # Clean up test files
    cleanup_test_files()
    
    return passed == total


if __name__ == "__main__":
    main()