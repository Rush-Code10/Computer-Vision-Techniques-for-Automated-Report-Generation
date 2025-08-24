"""
Demonstration script showing how to use the unified runner with accuracy evaluation.
This script demonstrates the key features implemented in Task 2.
"""

import os
from unified_runner import ChangeDetectionRunner


def demo_basic_usage():
    """Demonstrate basic usage of the unified runner."""
    print("🚀 DEMO: Basic Unified Runner Usage")
    print("=" * 50)
    
    # Check if test images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Orlando test images not found!")
        print("Please ensure orlando2010.png and orlando2023.png are in the current directory.")
        return
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    # Run all implementations
    print("Running all three implementations...")
    results = runner.run_all_implementations(img1_path, img2_path, min_area=100)
    
    print(f"\n✅ Successfully ran {len(results)} implementations")
    for result in results:
        print(f"  - {result.implementation_name}: {result.total_change_pixels:,} change pixels")


def demo_accuracy_evaluation():
    """Demonstrate accuracy evaluation features."""
    print("\n🔍 DEMO: Accuracy Evaluation Features")
    print("=" * 50)
    
    # Check if test images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Orlando test images not found!")
        return
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    # Run implementations
    print("Running implementations for accuracy evaluation...")
    results = runner.run_all_implementations(img1_path, img2_path, min_area=150)
    
    if len(results) < 2:
        print("❌ Need at least 2 results for accuracy evaluation")
        return
    
    # Perform accuracy evaluation
    print("\n📊 Performing accuracy evaluation...")
    accuracy_report = runner.evaluate_accuracy(results)
    
    # Print summary
    runner.evaluator.print_accuracy_summary(accuracy_report)
    
    # Show key metrics
    if "inter_method_agreement" in accuracy_report:
        agreement = accuracy_report["inter_method_agreement"]
        print(f"\n🎯 Key Findings:")
        print(f"  - Agreement Level: {agreement['agreement_level']}")
        print(f"  - Mean IoU between methods: {agreement['overall_agreement']['mean_iou']:.3f}")
        
        consensus = agreement["consensus_analysis"]
        print(f"  - Methods fully agree on {consensus['full_agreement_percentage']:.1f}% of pixels")


def demo_individual_implementation():
    """Demonstrate running individual implementations."""
    print("\n⚡ DEMO: Individual Implementation")
    print("=" * 50)
    
    # Check if test images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Orlando test images not found!")
        return
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    # Run just the basic implementation
    print("Running only the basic computer vision implementation...")
    result = runner.run_implementation('basic', img1_path, img2_path, min_area=200)
    
    print(f"\n✅ Basic CV Results:")
    print(f"  - Processing time: {result.processing_time:.3f} seconds")
    print(f"  - Change pixels: {result.total_change_pixels:,}")
    print(f"  - Change regions: {result.num_change_regions}")
    print(f"  - Average confidence: {result.average_confidence:.3f}")


def demo_comparison_analysis():
    """Demonstrate comparison analysis features."""
    print("\n📈 DEMO: Comparison Analysis")
    print("=" * 50)
    
    # Check if test images exist
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not (os.path.exists(img1_path) and os.path.exists(img2_path)):
        print("❌ Orlando test images not found!")
        return
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    # Run implementations
    print("Running implementations for comparison...")
    results = runner.run_all_implementations(img1_path, img2_path, min_area=100)
    
    if len(results) < 2:
        print("❌ Need at least 2 results for comparison")
        return
    
    # Generate comparison
    comparison = runner.compare_results(results)
    
    # Print detailed comparison
    runner.print_comparison(comparison)
    
    # Show additional insights
    print(f"\n💡 Additional Insights:")
    metrics = comparison['metrics_comparison']
    print(f"  - Change detection variance: {metrics['change_pixels']['std_dev']:,.0f} pixels")
    print(f"  - Speed difference: {metrics['processing_time']['slowest'] - metrics['processing_time']['fastest']:.3f} seconds")
    
    if metrics['confidence']['average'] > 0:
        print(f"  - Average confidence across methods: {metrics['confidence']['average']:.3f}")


def main():
    """Run all demonstrations."""
    print("🎯 UNIFIED RUNNER WITH ACCURACY EVALUATION - DEMO")
    print("=" * 60)
    print("This demo shows the key features implemented in Task 2:")
    print("1. Unified runner for all implementations")
    print("2. Basic accuracy evaluation with inter-method agreement")
    print("3. Comprehensive comparison and analysis")
    print("=" * 60)
    
    try:
        # Demo 1: Basic usage
        demo_basic_usage()
        
        # Demo 2: Accuracy evaluation
        demo_accuracy_evaluation()
        
        # Demo 3: Individual implementation
        demo_individual_implementation()
        
        # Demo 4: Comparison analysis
        demo_comparison_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Task 2 implementation includes:")
        print("✅ Main script that runs all three implementations")
        print("✅ Basic accuracy metrics (precision, recall, F1-score)")
        print("✅ Inter-method agreement analysis")
        print("✅ Comprehensive evaluation and reporting")
        print("✅ Command-line interface with accuracy evaluation")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()