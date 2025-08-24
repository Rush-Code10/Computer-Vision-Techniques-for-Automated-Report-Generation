"""
Test script for visualization components.
Tests the new standardized visualization functionality.
"""

import os
import numpy as np
from datetime import datetime
from typing import List

from standardized_data_models import ChangeDetectionResult, ChangeRegion, AccuracyMetrics
from visualization_components import VisualizationComponents


def create_mock_result(name: str, processing_time: float, change_pixels: int, num_regions: int) -> ChangeDetectionResult:
    """Create a mock change detection result for testing."""
    # Create mock change mask
    change_mask = np.random.randint(0, 2, (500, 500)) * 255
    
    # Create mock confidence map
    confidence_map = np.random.rand(500, 500)
    
    # Create mock change regions
    regions = []
    for i in range(num_regions):
        region = ChangeRegion(
            id=i,
            bbox=(np.random.randint(0, 400), np.random.randint(0, 400), 50, 50),
            area_pixels=np.random.randint(100, 1000),
            centroid=(np.random.randint(50, 450), np.random.randint(50, 450)),
            confidence=np.random.rand()
        )
        regions.append(region)
    
    return ChangeDetectionResult(
        implementation_name=name,
        version="1.0",
        timestamp=datetime.now(),
        processing_time=processing_time,
        change_mask=change_mask,
        confidence_map=confidence_map,
        change_regions=regions,
        total_change_pixels=change_pixels,
        num_change_regions=num_regions,
        average_confidence=np.mean([r.confidence for r in regions]) if regions else 0.0,
        image_dimensions=(500, 500),
        input_images=("test1.png", "test2.png"),
        parameters={"threshold": 0.5, "min_area": 100}
    )


def create_mock_accuracy_metrics(precision: float, recall: float) -> AccuracyMetrics:
    """Create mock accuracy metrics for testing."""
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = (precision * recall) / (precision + recall - precision * recall) if (precision + recall - precision * recall) > 0 else 0.0
    
    return AccuracyMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        iou=iou,
        accuracy=(precision + recall) / 2,
        specificity=0.9,
        true_positives=100,
        false_positives=20,
        true_negatives=800,
        false_negatives=30
    )


def test_visualization_components():
    """Test all visualization components."""
    print("Testing Visualization Components...")
    
    # Create test directory
    test_dir = "test_visualizations"
    os.makedirs(test_dir, exist_ok=True)
    
    # Initialize visualization components
    viz = VisualizationComponents()
    
    # Create mock data
    results = [
        create_mock_result("Basic Computer Vision", 0.123, 15000, 25),
        create_mock_result("Advanced Computer Vision", 0.456, 18500, 32),
        create_mock_result("Deep Learning Inspired", 0.789, 12000, 18)
    ]
    
    accuracy_data = {
        "Basic Computer Vision": create_mock_accuracy_metrics(0.85, 0.78),
        "Advanced Computer Vision": create_mock_accuracy_metrics(0.92, 0.81),
        "Deep Learning Inspired": create_mock_accuracy_metrics(0.88, 0.85)
    }
    
    print(f"Created {len(results)} mock results for testing")
    
    # Test 1: Individual change mask visualization
    print("\n1. Testing change mask visualization...")
    try:
        for result in results:
            viz_path = viz.create_change_mask_visualization(
                result, 
                save_path=os.path.join(test_dir, f"{result.implementation_name.lower().replace(' ', '_')}_mask.png")
            )
            print(f"   ✓ Created: {viz_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: Accuracy comparison plot
    print("\n2. Testing accuracy comparison plot...")
    try:
        acc_path = viz.create_accuracy_comparison_plot(
            accuracy_data,
            save_path=os.path.join(test_dir, "accuracy_comparison.png")
        )
        print(f"   ✓ Created: {acc_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: Performance comparison chart
    print("\n3. Testing performance comparison chart...")
    try:
        perf_path = viz.create_performance_comparison_chart(
            results,
            save_path=os.path.join(test_dir, "performance_comparison.png")
        )
        print(f"   ✓ Created: {perf_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: Change area statistics chart
    print("\n4. Testing change area statistics chart...")
    try:
        stats_path = viz.create_change_area_statistics_chart(
            results,
            save_path=os.path.join(test_dir, "change_area_stats.png")
        )
        print(f"   ✓ Created: {stats_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Comprehensive dashboard
    print("\n5. Testing comprehensive dashboard...")
    try:
        dashboard_path = viz.create_comprehensive_comparison_dashboard(
            results,
            accuracy_data,
            save_path=os.path.join(test_dir, "comprehensive_dashboard.png")
        )
        print(f"   ✓ Created: {dashboard_path}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print(f"\n✅ All visualization tests completed!")
    print(f"📁 Test outputs saved in: {test_dir}/")
    
    # List all created files
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        print(f"\n📊 Generated {len(files)} visualization files:")
        for file in sorted(files):
            print(f"   • {file}")


def test_color_consistency():
    """Test that colors are consistent across visualizations."""
    print("\n🎨 Testing color consistency...")
    
    viz = VisualizationComponents()
    
    # Check that color scheme is properly defined
    expected_colors = ['change', 'no_change', 'background', 'primary', 'secondary', 'accent']
    
    for color_name in expected_colors:
        if color_name in viz.colors:
            print(f"   ✓ {color_name}: {viz.colors[color_name]}")
        else:
            print(f"   ✗ Missing color: {color_name}")
    
    print("   ✅ Color scheme validation complete")


if __name__ == "__main__":
    print("=" * 60)
    print("VISUALIZATION COMPONENTS TEST SUITE")
    print("=" * 60)
    
    test_color_consistency()
    test_visualization_components()
    
    print("\n" + "=" * 60)
    print("TEST SUITE COMPLETED")
    print("=" * 60)