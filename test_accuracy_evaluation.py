"""
Test script for accuracy evaluation functionality.
Tests both ground truth evaluation and inter-method agreement analysis.
"""

import numpy as np
import os
from datetime import datetime
from unified_runner import ChangeDetectionRunner
from accuracy_evaluator import AccuracyEvaluator
from standardized_data_models import ChangeDetectionResult, ChangeRegion


def create_synthetic_ground_truth(shape=(500, 500)):
    """Create a synthetic ground truth mask for testing."""
    gt = np.zeros(shape, dtype=np.uint8)
    
    # Add some rectangular change regions
    gt[100:200, 150:250] = 255  # Large rectangle
    gt[300:350, 100:180] = 255  # Medium rectangle
    gt[400:450, 350:400] = 255  # Small rectangle
    
    return gt


def create_synthetic_result(name, shape=(500, 500), noise_level=0.1):
    """Create a synthetic change detection result for testing."""
    # Create a mask similar to ground truth but with some noise
    mask = np.zeros(shape, dtype=np.uint8)
    
    if name == "Perfect":
        # Perfect match with ground truth
        mask[100:200, 150:250] = 255
        mask[300:350, 100:180] = 255
        mask[400:450, 350:400] = 255
    elif name == "Good":
        # Good match with slight variations
        mask[105:195, 155:245] = 255  # Slightly smaller
        mask[300:350, 100:180] = 255  # Same
        mask[405:445, 355:395] = 255  # Slightly smaller
        # Add some false positive
        mask[50:70, 50:70] = 255
    elif name == "Poor":
        # Poor match with many differences
        mask[120:180, 170:230] = 255  # Much smaller
        mask[310:340, 110:170] = 255  # Smaller
        # Missing the third region
        # Add several false positives
        mask[50:80, 50:80] = 255
        mask[200:230, 300:330] = 255
        mask[450:480, 200:230] = 255
    
    # Add random noise
    if noise_level > 0:
        noise = np.random.random(shape) < noise_level
        mask[noise] = 255 - mask[noise]  # Flip pixels
    
    # Create change regions from mask
    import cv2
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    change_regions = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= 50:  # Minimum area threshold
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            region = ChangeRegion(
                id=i,
                bbox=(x, y, w, h),
                area_pixels=area,
                centroid=(centroids[i][0], centroids[i][1]),
                confidence=np.random.uniform(0.7, 1.0)
            )
            change_regions.append(region)
    
    # Create result object
    result = ChangeDetectionResult(
        implementation_name=name,
        version="1.0",
        timestamp=datetime.now(),
        processing_time=np.random.uniform(0.5, 2.0),
        change_mask=mask,
        change_regions=change_regions,
        total_change_area=float(np.sum(mask > 0)),
        total_change_pixels=int(np.sum(mask > 0)),
        num_change_regions=len(change_regions),
        average_confidence=np.mean([r.confidence for r in change_regions]) if change_regions else 0.0,
        parameters={"synthetic": True, "noise_level": noise_level},
        input_images=("synthetic1.png", "synthetic2.png"),
        image_dimensions=shape
    )
    
    return result


def test_ground_truth_evaluation():
    """Test accuracy evaluation with ground truth."""
    print("=" * 60)
    print("TESTING GROUND TRUTH EVALUATION")
    print("=" * 60)
    
    # Create synthetic data
    shape = (500, 500)
    ground_truth = create_synthetic_ground_truth(shape)
    
    # Create synthetic results with different quality levels
    results = [
        create_synthetic_result("Perfect", shape, noise_level=0.0),
        create_synthetic_result("Good", shape, noise_level=0.05),
        create_synthetic_result("Poor", shape, noise_level=0.15)
    ]
    
    # Test evaluation
    evaluator = AccuracyEvaluator()
    
    print("Ground Truth Evaluation Results:")
    print("-" * 40)
    
    for result in results:
        try:
            metrics = evaluator.evaluate_with_ground_truth(result, ground_truth)
            print(f"\n{result.implementation_name}:")
            print(f"  Precision: {metrics.precision:.3f}")
            print(f"  Recall:    {metrics.recall:.3f}")
            print(f"  F1-Score:  {metrics.f1_score:.3f}")
            print(f"  IoU:       {metrics.iou:.3f}")
            print(f"  Accuracy:  {metrics.accuracy:.3f}")
            print(f"  TP: {metrics.true_positives}, FP: {metrics.false_positives}")
            print(f"  TN: {metrics.true_negatives}, FN: {metrics.false_negatives}")
        except Exception as e:
            print(f"❌ {result.implementation_name} failed: {str(e)}")
    
    return results, ground_truth


def test_inter_method_agreement():
    """Test inter-method agreement analysis."""
    print("\n" + "=" * 60)
    print("TESTING INTER-METHOD AGREEMENT")
    print("=" * 60)
    
    # Create synthetic results with varying agreement levels
    shape = (400, 400)
    results = [
        create_synthetic_result("Method_A", shape, noise_level=0.05),
        create_synthetic_result("Method_B", shape, noise_level=0.08),
        create_synthetic_result("Method_C", shape, noise_level=0.12)
    ]
    
    # Test agreement analysis
    evaluator = AccuracyEvaluator()
    
    try:
        agreement_analysis = evaluator.evaluate_inter_method_agreement(results)
        
        print("Inter-Method Agreement Analysis:")
        print("-" * 40)
        print(f"Agreement Level: {agreement_analysis['agreement_level']}")
        print(f"Mean IoU: {agreement_analysis['overall_agreement']['mean_iou']:.3f}")
        print(f"Mean Jaccard: {agreement_analysis['overall_agreement']['mean_jaccard']:.3f}")
        
        print(f"\nPairwise Agreements:")
        for pair, metrics in agreement_analysis['pairwise_agreements'].items():
            print(f"  {pair}:")
            print(f"    IoU: {metrics['iou']:.3f}")
            print(f"    Jaccard: {metrics['jaccard_similarity']:.3f}")
            print(f"    Pixel Agreement: {metrics['pixel_agreement']:.3f}")
        
        consensus = agreement_analysis['consensus_analysis']
        print(f"\nConsensus Analysis:")
        print(f"  Full Agreement: {consensus['full_agreement_percentage']:.1f}% of pixels")
        print(f"  Partial Agreement: {consensus['partial_agreement_pixels']:,} pixels")
        
    except Exception as e:
        print(f"❌ Agreement analysis failed: {str(e)}")
    
    return results


def test_comprehensive_evaluation():
    """Test comprehensive accuracy evaluation."""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    # Create test data
    shape = (300, 300)
    ground_truth = create_synthetic_ground_truth(shape)
    results = [
        create_synthetic_result("CV_Basic", shape, noise_level=0.1),
        create_synthetic_result("CV_Advanced", shape, noise_level=0.08),
        create_synthetic_result("Deep_Learning", shape, noise_level=0.06)
    ]
    
    # Test comprehensive evaluation
    evaluator = AccuracyEvaluator()
    
    try:
        report = evaluator.generate_accuracy_report(results, ground_truth)
        evaluator.print_accuracy_summary(report)
        
        print(f"\n📄 Full report contains {len(report)} sections")
        print(f"Report sections: {list(report.keys())}")
        
    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {str(e)}")


def test_with_real_data():
    """Test with real Orlando airport data if available."""
    print("\n" + "=" * 60)
    print("TESTING WITH REAL DATA")
    print("=" * 60)
    
    # Check if Orlando images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Orlando test images not found, skipping real data test")
        return
    
    print("✅ Found Orlando test images, running real evaluation...")
    
    # Create runner and run implementations
    runner = ChangeDetectionRunner()
    
    try:
        results = runner.run_all_implementations(img1_path, img2_path, min_area=100)
        
        if len(results) >= 2:
            print(f"\n📊 Running accuracy evaluation on {len(results)} real results...")
            accuracy_report = runner.evaluate_accuracy(results)
            runner.evaluator.print_accuracy_summary(accuracy_report)
        else:
            print("❌ Need at least 2 results for agreement analysis")
            
    except Exception as e:
        print(f"❌ Real data test failed: {str(e)}")


def main():
    """Run all accuracy evaluation tests."""
    print("🧪 ACCURACY EVALUATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Test 1: Ground truth evaluation
        results, gt = test_ground_truth_evaluation()
        
        # Test 2: Inter-method agreement
        test_inter_method_agreement()
        
        # Test 3: Comprehensive evaluation
        test_comprehensive_evaluation()
        
        # Test 4: Real data (if available)
        test_with_real_data()
        
        print("\n" + "=" * 60)
        print("🎉 ALL ACCURACY EVALUATION TESTS COMPLETED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()