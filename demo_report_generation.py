"""
Demo script showcasing the standardized report generation functionality.
This script demonstrates how to generate individual, comparison, and executive summary reports.
"""

import os
from unified_runner import ChangeDetectionRunner


def demo_report_generation():
    """Demonstrate the report generation capabilities."""
    print("=" * 60)
    print("STANDARDIZED REPORT GENERATION DEMO")
    print("=" * 60)
    
    # Check if required images exist
    img1 = "orlando2010.png"
    img2 = "orlando2023.png"
    
    if not os.path.exists(img1) or not os.path.exists(img2):
        print(f"❌ Required images not found: {img1}, {img2}")
        print("Please ensure the Orlando airport images are in the current directory.")
        return
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    print(f"📸 Analyzing images: {img1} and {img2}")
    print("🔄 Running all change detection implementations...")
    
    # Run all implementations
    results = runner.run_all_implementations(img1, img2, min_area=100)
    
    if not results:
        print("❌ No results to process")
        return
    
    print(f"✅ Successfully processed {len(results)} implementations")
    
    # Generate comparison if multiple results
    comparison = {}
    if len(results) > 1:
        comparison = runner.compare_results(results)
        runner.print_comparison(comparison)
    
    # Perform accuracy evaluation
    print("\n🔍 Performing accuracy evaluation...")
    accuracy_report = runner.evaluate_accuracy(results)
    runner.evaluator.print_accuracy_summary(accuracy_report)
    
    # Generate reports
    print("\n📄 Generating standardized reports...")
    reports_dir = "demo_reports"
    generated_reports = runner.generate_reports(results, accuracy_report, reports_dir)
    
    if generated_reports:
        print(f"\n🎉 Successfully generated {len(generated_reports)} report(s):")
        print("-" * 50)
        
        for report_type, path in generated_reports.items():
            file_size = os.path.getsize(path) / 1024  # Size in KB
            print(f"📋 {report_type.replace('_', ' ').title()}")
            print(f"   Path: {path}")
            print(f"   Size: {file_size:.1f} KB")
            print()
        
        print("📁 All reports are saved in the 'demo_reports' directory")
        print("💡 You can open these PDF files to view the detailed analysis")
        
        # Show report contents summary
        print("\n📊 Report Contents Summary:")
        print("-" * 30)
        
        individual_count = sum(1 for key in generated_reports.keys() if 'individual' in key)
        has_comparison = 'comparison' in generated_reports
        has_executive = 'executive_summary' in generated_reports
        
        print(f"• Individual Method Reports: {individual_count}")
        if has_comparison:
            print("• Comparison Report: ✅ (Side-by-side method analysis)")
        if has_executive:
            print("• Executive Summary: ✅ (High-level findings and recommendations)")
        
        print("\n🔍 Each report includes:")
        print("  - Method descriptions and technical details")
        print("  - Results summary with change statistics")
        print("  - Visualizations and charts")
        print("  - Accuracy metrics (when available)")
        print("  - Recommendations and next steps")
        
    else:
        print("❌ No reports were generated")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_report_generation()