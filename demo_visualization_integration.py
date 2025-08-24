"""
Demonstration of integrated visualization components with the reporting system.
Shows how the new visualization features work with real change detection results.
"""

import os
import json
import numpy as np
from datetime import datetime

from unified_runner import ChangeDetectionRunner
from report_generator import ReportGenerator
from accuracy_evaluator import AccuracyEvaluator
from visualization_components import VisualizationComponents


def demo_visualization_integration():
    """Demonstrate the complete visualization integration."""
    print("=" * 70)
    print("VISUALIZATION INTEGRATION DEMONSTRATION")
    print("=" * 70)
    
    # Initialize components
    runner = ChangeDetectionRunner()
    report_gen = ReportGenerator()
    evaluator = AccuracyEvaluator()
    viz_components = VisualizationComponents()
    
    # Run all implementations
    print("🚀 Running all change detection implementations...")
    img1_path = "orlando2010.png"
    img2_path = "orlando2023.png"
    
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"❌ Input images not found: {img1_path}, {img2_path}")
        print("   Using mock data for demonstration...")
        # Create mock results for demonstration
        from test_visualization_components import create_mock_result
        results = [
            create_mock_result("Basic Computer Vision", 0.123, 15000, 25),
            create_mock_result("Advanced Computer Vision", 0.456, 18500, 32),
            create_mock_result("Deep Learning Inspired", 0.789, 12000, 18)
        ]
    else:
        results = runner.run_all_implementations(img1_path, img2_path)
    
    if not results:
        print("❌ No results generated. Please check input images.")
        return
    
    print(f"✅ Generated {len(results)} results")
    
    # Generate accuracy evaluation
    print("\n📊 Generating accuracy evaluation...")
    accuracy_report = evaluator.generate_accuracy_report(results)
    
    # Create output directory for demo
    demo_dir = "demo_visualization_integration"
    os.makedirs(demo_dir, exist_ok=True)
    
    print(f"\n🎨 Creating individual visualizations...")
    
    # 1. Individual change mask visualizations
    individual_viz_paths = []
    for result in results:
        viz_path = viz_components.create_change_mask_visualization(
            result,
            save_path=os.path.join(demo_dir, f"{result.implementation_name.lower().replace(' ', '_')}_visualization.png")
        )
        individual_viz_paths.append(viz_path)
        print(f"   ✓ {result.implementation_name}: {os.path.basename(viz_path)}")
    
    # 2. Performance comparison chart
    print(f"\n📈 Creating performance comparison chart...")
    perf_path = viz_components.create_performance_comparison_chart(
        results,
        save_path=os.path.join(demo_dir, "performance_comparison.png")
    )
    print(f"   ✓ Performance chart: {os.path.basename(perf_path)}")
    
    # 3. Change area statistics
    print(f"\n📊 Creating change area statistics...")
    stats_path = viz_components.create_change_area_statistics_chart(
        results,
        save_path=os.path.join(demo_dir, "change_area_statistics.png")
    )
    print(f"   ✓ Statistics chart: {os.path.basename(stats_path)}")
    
    # 4. Accuracy comparison (if ground truth evaluation available)
    if "ground_truth_evaluation" in accuracy_report:
        print(f"\n🎯 Creating accuracy comparison...")
        from standardized_data_models import AccuracyMetrics
        
        accuracy_data = {}
        for method, metrics in accuracy_report["ground_truth_evaluation"].items():
            if "error" not in metrics:
                accuracy_data[method] = AccuracyMetrics(
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1_score=metrics["f1_score"],
                    iou=metrics["iou"],
                    accuracy=metrics["accuracy"],
                    specificity=metrics.get("specificity", 0.0)
                )
        
        if accuracy_data:
            acc_path = viz_components.create_accuracy_comparison_plot(
                accuracy_data,
                save_path=os.path.join(demo_dir, "accuracy_comparison.png")
            )
            print(f"   ✓ Accuracy chart: {os.path.basename(acc_path)}")
    
    # 5. Comprehensive dashboard
    print(f"\n🎛️ Creating comprehensive dashboard...")
    dashboard_path = viz_components.create_comprehensive_comparison_dashboard(
        results,
        save_path=os.path.join(demo_dir, "comprehensive_dashboard.png")
    )
    print(f"   ✓ Dashboard: {os.path.basename(dashboard_path)}")
    
    # 6. Generate reports with new visualizations
    print(f"\n📄 Generating reports with integrated visualizations...")
    
    # Individual reports
    individual_reports = []
    for result in results:
        report_path = report_gen.generate_individual_report(
            result,
            output_path=os.path.join(demo_dir, f"{result.implementation_name.lower().replace(' ', '_')}_report.pdf")
        )
        individual_reports.append(report_path)
        print(f"   ✓ Individual report: {os.path.basename(report_path)}")
    
    # Comparison report
    comparison_path = report_gen.generate_comparison_report(
        results,
        accuracy_report,
        output_path=os.path.join(demo_dir, "comparison_report.pdf")
    )
    print(f"   ✓ Comparison report: {os.path.basename(comparison_path)}")
    
    # Executive summary
    exec_path = report_gen.generate_executive_summary_report(
        results,
        accuracy_report,
        output_path=os.path.join(demo_dir, "executive_summary.pdf")
    )
    print(f"   ✓ Executive summary: {os.path.basename(exec_path)}")
    
    # 7. Save configuration and metadata
    print(f"\n💾 Saving metadata and configuration...")
    
    # Save accuracy report
    with open(os.path.join(demo_dir, "accuracy_evaluation.json"), 'w') as f:
        json.dump(accuracy_report, f, indent=2, default=str)
    
    # Save results summary
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "num_methods": len(results),
        "methods": [r.implementation_name for r in results],
        "total_change_pixels": [r.total_change_pixels for r in results],
        "processing_times": [r.processing_time for r in results],
        "num_regions": [r.num_change_regions for r in results],
        "visualization_files": {
            "individual_visualizations": [os.path.basename(p) for p in individual_viz_paths],
            "performance_chart": os.path.basename(perf_path),
            "statistics_chart": os.path.basename(stats_path),
            "dashboard": os.path.basename(dashboard_path)
        },
        "report_files": {
            "individual_reports": [os.path.basename(p) for p in individual_reports],
            "comparison_report": os.path.basename(comparison_path),
            "executive_summary": os.path.basename(exec_path)
        }
    }
    
    with open(os.path.join(demo_dir, "demo_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"   ✓ Metadata saved")
    
    # 8. Print summary
    print(f"\n" + "=" * 70)
    print("DEMONSTRATION SUMMARY")
    print("=" * 70)
    
    print(f"📁 Output directory: {demo_dir}/")
    print(f"🔢 Methods analyzed: {len(results)}")
    print(f"🎨 Visualizations created: {len(individual_viz_paths) + 4}")  # +4 for comparison charts
    print(f"📄 Reports generated: {len(individual_reports) + 2}")  # +2 for comparison and executive
    
    # List all files
    all_files = os.listdir(demo_dir)
    print(f"\n📊 Generated {len(all_files)} files:")
    
    # Group files by type
    visualizations = [f for f in all_files if f.endswith('.png')]
    reports = [f for f in all_files if f.endswith('.pdf')]
    data_files = [f for f in all_files if f.endswith('.json')]
    
    if visualizations:
        print(f"\n🎨 Visualizations ({len(visualizations)}):")
        for viz in sorted(visualizations):
            print(f"   • {viz}")
    
    if reports:
        print(f"\n📄 Reports ({len(reports)}):")
        for report in sorted(reports):
            print(f"   • {report}")
    
    if data_files:
        print(f"\n💾 Data Files ({len(data_files)}):")
        for data in sorted(data_files):
            print(f"   • {data}")
    
    print(f"\n✅ Visualization integration demonstration completed successfully!")
    print(f"🎉 All components working together with consistent styling and formatting!")


def print_visualization_features():
    """Print information about the new visualization features."""
    print("\n" + "=" * 70)
    print("NEW VISUALIZATION FEATURES")
    print("=" * 70)
    
    features = [
        "🎨 Consistent Color Scheme: Standardized colors across all visualizations",
        "📊 Change Mask Visualization: Enhanced change detection display with legends",
        "📈 Performance Comparison Charts: Processing time, efficiency, and statistics",
        "🎯 Accuracy Comparison Plots: Bar charts and radar charts for metrics",
        "📋 Change Area Statistics: Detailed analysis of detected changes",
        "🎛️ Comprehensive Dashboard: All-in-one comparison view",
        "🔄 Integrated Report Generation: Seamless integration with PDF reports",
        "📐 Standardized Legends: Consistent legends and annotations",
        "🎨 Professional Styling: Publication-ready visualizations",
        "⚡ Efficient Processing: Optimized for large datasets"
    ]
    
    for feature in features:
        print(f"   {feature}")
    
    print(f"\n✨ All visualizations follow the same design principles:")
    print(f"   • Consistent color palette")
    print(f"   • Professional typography")
    print(f"   • Clear legends and labels")
    print(f"   • High-resolution output")
    print(f"   • Accessibility-friendly design")


if __name__ == "__main__":
    print_visualization_features()
    demo_visualization_integration()