"""
Test script to verify that all implementations work correctly
and return results in the standardized format.
"""

import os
import numpy as np
from implementation_extractors import (
    basic_cv_implementation,
    advanced_cv_implementation, 
    deep_learning_implementation
)
from standardized_data_models import ChangeDetectionResult


def test_implementation(impl_func, name, img1_path, img2_path):
    """Test a single implementation and print results."""
    print(f"\n{'='*50}")
    print(f"Testing {name}")
    print(f"{'='*50}")
    
    try:
        # Run the implementation
        result = impl_func(img1_path, img2_path)
        
        # Verify it returns the correct type
        assert isinstance(result, ChangeDetectionResult), f"Expected ChangeDetectionResult, got {type(result)}"
        
        # Print key metrics
        print(f"Implementation: {result.implementation_name}")
        print(f"Version: {result.version}")
        print(f"Processing time: {result.processing_time:.3f} seconds")
        print(f"Total change pixels: {result.total_change_pixels:,}")
        print(f"Total change area: {result.total_change_area:.2f}")
        print(f"Number of change regions: {result.num_change_regions}")
        print(f"Average confidence: {result.average_confidence:.3f}")
        print(f"Image dimensions: {result.image_dimensions}")
        print(f"Input images: {result.input_images}")
        
        # Verify mask properties
        if result.change_mask is not None:
            print(f"Change mask shape: {result.change_mask.shape}")
            print(f"Change mask dtype: {result.change_mask.dtype}")
            print(f"Change mask unique values: {np.unique(result.change_mask)}")
        
        # Verify confidence map if present
        if result.confidence_map is not None:
            print(f"Confidence map shape: {result.confidence_map.shape}")
            print(f"Confidence map range: [{result.confidence_map.min():.3f}, {result.confidence_map.max():.3f}]")
        
        # Print some region details
        if result.change_regions:
            print(f"\nFirst few change regions:")
            for i, region in enumerate(result.change_regions[:3]):
                print(f"  Region {region.id}: bbox={region.bbox}, area={region.area_pixels}, conf={region.confidence:.3f}")
        
        print(f"✅ {name} completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ {name} failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("Testing Standardized Change Detection Implementations")
    print("=" * 60)
    
    # Check if test images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not os.path.exists(img1_path):
        print(f"❌ Test image {img1_path} not found!")
        return
    
    if not os.path.exists(img2_path):
        print(f"❌ Test image {img2_path} not found!")
        return
    
    print(f"Using test images: {img1_path} and {img2_path}")
    
    # Test all implementations
    implementations = [
        (basic_cv_implementation, "Basic Computer Vision"),
        (advanced_cv_implementation, "Advanced Computer Vision"),
        (deep_learning_implementation, "Deep Learning Inspired")
    ]
    
    results = []
    for impl_func, name in implementations:
        success = test_implementation(impl_func, name, img1_path, img2_path)
        results.append((name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} implementations passed")
    
    if total_passed == len(results):
        print("🎉 All implementations are working correctly!")
    else:
        print("⚠️  Some implementations need attention.")


if __name__ == "__main__":
    main()