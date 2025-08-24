"""
Test script for the report generator module.
Tests individual report generation, comparison reports, and executive summaries.
"""

import os
import numpy as np
from datetime import datetime
from standardized_data_models import ChangeDetectionResult, ChangeRegion, AccuracyMetrics
from report_generator import ReportGenerator


def create_sample_result(name: str, change_pixels: int, regions: int, processing_time: float) -> ChangeDetectionResult:
    """Create a sample change detection result for testing."""
    # Create sample change mask
    change_mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8) * 255
    change_mask[change_mask > 0] = np.random.randint(1, 256, size=np.sum(change_mask > 0))
    
    # Create sample regions
    change_regions = []
    for i in range(regions):
        region = ChangeRegion(
            id=i,
            bbox=(i*50, i*50, 100, 100),
            area_pixels=np.random.randint(100, 1000),
            centroid=(i*50 + 50, i*50 + 50),
            confidence=np.random.uniform(0.5, 1.0),
            change_type="construction" if i % 2 == 0 else "expansion"
        )
        change_regions.append(region)
    
    # Create confidence map for some methods
    confidence_map = None
    if "Advanced" in name or "Deep Learning" in name:
        confidence_map = np.random.uniform(0, 1, (500, 500))
    
    result = ChangeDetectionResult(
        implementation_name=name,
        version="1.0.0",
        timestamp=datetime.now(),
        processing_time=processing_time,
        change_mask=change_mask,
        confidence_map=confidence_map,
        change_regions=change_regions,
        total_change_pixels=change_pixels,
        num_change_regions=regions,
        average_confidence=np.mean([r.confidence for r in change_regions]) if change_regions else 0.0,
        parameters={"min_area": 100, "threshold": 0.5},
        input_images=("orlando2010.png", "orlando2023.png"),
        image_dimensions=(500, 500)
    )
    
    return result


def create_sample_accuracy_metrics() -> AccuracyMetrics:
    """Create sample accuracy metrics for testing."""
    return AccuracyMetrics(
        precision=0.85,
        recall=0.78,
        f1_score=0.81,
        iou=0.68,
        accuracy=0.92,
        specificity=0.95,
        true_positives=1250,
        false_positives=220,
        true_negatives=18500,
        false_negatives=350
    )


def create_sample_accuracy_report(results: list) -> dict:
    """Create a sample accuracy report for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "num_methods": len(results),
        "method_names": [r.implementation_name for r in results],
        "ground_truth_evaluation": {
            "Basic Computer Vision": {
                "precision": 0.82,
                "recall": 0.75,
                "f1_score": 0.78,
                "iou": 0.64,
                "accuracy": 0.89,
                "specificity": 0.93
            },
            "Advanced Computer Vision": {
                "precision": 0.88,
                "recall": 0.81,
                "f1_score": 0.84,
                "iou": 0.72,
                "accuracy": 0.94,
                "specificity": 0.96
            },
            "Deep Learning Inspired": {
                "precision": 0.91,
                "recall": 0.85,
                "f1_score": 0.88,
                "iou": 0.78,
                "accuracy": 0.96,
                "specificity": 0.97
            }
        },
        "inter_method_agreement": {
            "num_methods": len(results),
            "method_names": [r.implementation_name for r in results],
            "overall_agreement": {
                "mean_iou": 0.68,
                "mean_jaccard": 0.71,
                "std_iou": 0.12,
                "std_jaccard": 0.09
            },
            "agreement_level": "Medium",
            "consensus_analysis": {
                "full_agreement_percentage": 72.5,
                "partial_agreement_pixels": 45000,
                "consensus_mask_mean": 0.15
            }
        }
    }


def test_individual_report():
    """Test individual report generation."""
    print("Testing individual report generation...")
    
    # Create sample data
    result = create_sample_result("Basic Computer Vision", 15000, 25, 2.45)
    accuracy_metrics = create_sample_accuracy_metrics()
    
    # Generate report
    generator = ReportGenerator()
    report_path = generator.generate_individual_report(result, accuracy_metrics)
    
    if os.path.exists(report_path):
        print(f"✅ Individual report generated successfully: {report_path}")
        return True
    else:
        print(f"❌ Individual report generation failed")
        return False


def test_comparison_report():
    """Test comparison report generation."""
    print("Testing comparison report generation...")
    
    # Create sample data
    results = [
        create_sample_result("Basic Computer Vision", 15000, 25, 2.45),
        create_sample_result("Advanced Computer Vision", 18500, 32, 4.12),
        create_sample_result("Deep Learning Inspired", 22000, 28, 8.75)
    ]
    
    accuracy_report = create_sample_accuracy_report(results)
    
    # Generate report
    generator = ReportGenerator()
    report_path = generator.generate_comparison_report(results, accuracy_report)
    
    if os.path.exists(report_path):
        print(f"✅ Comparison report generated successfully: {report_path}")
        return True
    else:
        print(f"❌ Comparison report generation failed")
        return False


def test_executive_summary():
    """Test executive summary report generation."""
    print("Testing executive summary generation...")
    
    # Create sample data
    results = [
        create_sample_result("Basic Computer Vision", 15000, 25, 2.45),
        create_sample_result("Advanced Computer Vision", 18500, 32, 4.12),
        create_sample_result("Deep Learning Inspired", 22000, 28, 8.75)
    ]
    
    accuracy_report = create_sample_accuracy_report(results)
    
    # Generate report
    generator = ReportGenerator()
    report_path = generator.generate_executive_summary_report(results, accuracy_report)
    
    if os.path.exists(report_path):
        print(f"✅ Executive summary generated successfully: {report_path}")
        return True
    else:
        print(f"❌ Executive summary generation failed")
        return False


def test_report_generation_without_accuracy():
    """Test report generation without accuracy metrics."""
    print("Testing report generation without accuracy metrics...")
    
    # Create sample data
    result = create_sample_result("Basic Computer Vision", 15000, 25, 2.45)
    
    # Generate report without accuracy metrics
    generator = ReportGenerator()
    report_path = generator.generate_individual_report(result, accuracy_metrics=None)
    
    if os.path.exists(report_path):
        print(f"✅ Report without accuracy metrics generated successfully: {report_path}")
        return True
    else:
        print(f"❌ Report generation without accuracy metrics failed")
        return False


def main():
    """Run all report generator tests."""
    print("=" * 60)
    print("REPORT GENERATOR TESTS")
    print("=" * 60)
    
    # Ensure reports directory exists
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/visualizations", exist_ok=True)
    
    tests = [
        test_individual_report,
        test_comparison_report,
        test_executive_summary,
        test_report_generation_without_accuracy
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
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All report generator tests passed!")
    else:
        print(f"⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    main()