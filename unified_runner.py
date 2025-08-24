"""
Unified runner for all change detection implementations.
This script can run individual implementations or all implementations together,
providing standardized output and basic comparison capabilities.

Note: This module is now integrated with the new CLI system (cli.py) which provides
enhanced configuration management, logging, and workflow control. For new usage,
consider using cli.py instead of this module directly.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import cv2

from implementation_extractors import (
    basic_cv_implementation,
    advanced_cv_implementation,
    deep_learning_implementation
)
from standardized_data_models import ChangeDetectionResult
from accuracy_evaluator import AccuracyEvaluator
from report_generator import ReportGenerator


class ChangeDetectionRunner:
    """Main runner class for change detection implementations."""
    
    def __init__(self):
        self.implementations = {
            'basic': basic_cv_implementation,
            'advanced': advanced_cv_implementation,
            'deep_learning': deep_learning_implementation
        }
        self.evaluator = AccuracyEvaluator()
        self.report_generator = ReportGenerator()
    
    def run_implementation(self, name: str, img1_path: str, img2_path: str, **kwargs) -> ChangeDetectionResult:
        """Run a single implementation."""
        if name not in self.implementations:
            raise ValueError(f"Unknown implementation: {name}. Available: {list(self.implementations.keys())}")
        
        print(f"Running {name} implementation...")
        result = self.implementations[name](img1_path, img2_path, **kwargs)
        print(f"✅ {name} completed in {result.processing_time:.3f} seconds")
        return result
    
    def run_all_implementations(self, img1_path: str, img2_path: str, **kwargs) -> List[ChangeDetectionResult]:
        """Run all implementations and return results."""
        results = []
        
        print("Running all change detection implementations...")
        print("=" * 50)
        
        for name in self.implementations.keys():
            try:
                result = self.run_implementation(name, img1_path, img2_path, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
        
        return results
    
    def compare_results(self, results: List[ChangeDetectionResult]) -> Dict[str, Any]:
        """Compare results from multiple implementations."""
        if not results:
            return {}
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'num_implementations': len(results),
            'implementations': [],
            'metrics_comparison': {},
            'processing_times': {},
            'summary': {}
        }
        
        # Collect data for comparison
        processing_times = []
        change_pixels = []
        num_regions = []
        confidences = []
        
        for result in results:
            impl_data = {
                'name': result.implementation_name,
                'version': result.version,
                'processing_time': result.processing_time,
                'total_change_pixels': result.total_change_pixels,
                'num_change_regions': result.num_change_regions,
                'average_confidence': result.average_confidence,
                'parameters': result.parameters
            }
            comparison['implementations'].append(impl_data)
            
            processing_times.append(result.processing_time)
            change_pixels.append(result.total_change_pixels)
            num_regions.append(result.num_change_regions)
            confidences.append(result.average_confidence)
            comparison['processing_times'][result.implementation_name] = result.processing_time
        
        # Calculate comparison metrics
        comparison['metrics_comparison'] = {
            'processing_time': {
                'fastest': min(processing_times),
                'slowest': max(processing_times),
                'average': np.mean(processing_times)
            },
            'change_pixels': {
                'minimum': min(change_pixels),
                'maximum': max(change_pixels),
                'average': np.mean(change_pixels),
                'std_dev': np.std(change_pixels)
            },
            'num_regions': {
                'minimum': min(num_regions),
                'maximum': max(num_regions),
                'average': np.mean(num_regions)
            },
            'confidence': {
                'minimum': min(confidences) if confidences else 0,
                'maximum': max(confidences) if confidences else 0,
                'average': np.mean(confidences) if confidences else 0
            }
        }
        
        # Generate summary
        fastest_impl = results[processing_times.index(min(processing_times))].implementation_name
        most_changes = results[change_pixels.index(max(change_pixels))].implementation_name
        most_regions = results[num_regions.index(max(num_regions))].implementation_name
        
        comparison['summary'] = {
            'fastest_implementation': fastest_impl,
            'most_changes_detected': most_changes,
            'most_regions_detected': most_regions,
            'agreement_level': self._calculate_agreement(results)
        }
        
        return comparison
    
    def _calculate_agreement(self, results: List[ChangeDetectionResult]) -> str:
        """Calculate a simple agreement level between implementations."""
        if len(results) < 2:
            return "N/A"
        
        change_pixels = [r.total_change_pixels for r in results]
        cv = np.std(change_pixels) / np.mean(change_pixels) if np.mean(change_pixels) > 0 else 0
        
        if cv < 0.2:
            return "High"
        elif cv < 0.5:
            return "Medium"
        else:
            return "Low"
    
    def evaluate_accuracy(self, results: List[ChangeDetectionResult], 
                         ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate accuracy of results with optional ground truth.
        
        Args:
            results: List of change detection results
            ground_truth_path: Optional path to ground truth mask image
            
        Returns:
            Dictionary containing accuracy evaluation report
        """
        ground_truth = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            # Load ground truth image
            gt_img = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                ground_truth = gt_img
                print(f"Loaded ground truth from {ground_truth_path}")
            else:
                print(f"Warning: Could not load ground truth from {ground_truth_path}")
        
        # Generate accuracy report
        accuracy_report = self.evaluator.generate_accuracy_report(results, ground_truth)
        return accuracy_report
    
    def save_results(self, results: List[ChangeDetectionResult], comparison: Dict[str, Any], 
                    accuracy_report: Optional[Dict[str, Any]] = None, output_dir: str = "results"):
        """Save results, comparison, and accuracy evaluation to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison summary
        comparison_file = os.path.join(output_dir, "comparison_summary.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Save accuracy report if available
        if accuracy_report:
            accuracy_file = os.path.join(output_dir, "accuracy_evaluation.json")
            with open(accuracy_file, 'w') as f:
                json.dump(accuracy_report, f, indent=2, default=str)
        
        # Save individual results
        for result in results:
            # Create a serializable version of the result
            result_dict = {
                'implementation_name': result.implementation_name,
                'version': result.version,
                'timestamp': result.timestamp.isoformat(),
                'processing_time': result.processing_time,
                'total_change_area': result.total_change_area,
                'total_change_pixels': result.total_change_pixels,
                'num_change_regions': result.num_change_regions,
                'average_confidence': result.average_confidence,
                'parameters': result.parameters,
                'input_images': result.input_images,
                'image_dimensions': result.image_dimensions,
                'change_regions': [
                    {
                        'id': r.id,
                        'bbox': r.bbox,
                        'area_pixels': r.area_pixels,
                        'centroid': r.centroid,
                        'confidence': r.confidence,
                        'change_type': r.change_type
                    } for r in result.change_regions
                ]
            }
            
            # Save individual result
            result_file = os.path.join(output_dir, f"{result.implementation_name.lower().replace(' ', '_')}_result.json")
            with open(result_file, 'w') as f:
                json.dump(result_dict, f, indent=2, default=str)
            
            # Save change mask
            mask_file = os.path.join(output_dir, f"{result.implementation_name.lower().replace(' ', '_')}_mask.npy")
            np.save(mask_file, result.change_mask)
            
            # Save confidence map if available
            if result.confidence_map is not None:
                conf_file = os.path.join(output_dir, f"{result.implementation_name.lower().replace(' ', '_')}_confidence.npy")
                np.save(conf_file, result.confidence_map)
        
        print(f"Results saved to {output_dir}/")
    
    def generate_reports(self, results: List[ChangeDetectionResult], 
                        accuracy_report: Optional[Dict[str, Any]] = None,
                        output_dir: str = "reports") -> Dict[str, str]:
        """
        Generate standardized PDF reports for results.
        
        Args:
            results: List of change detection results
            accuracy_report: Optional accuracy evaluation report
            output_dir: Output directory for reports
            
        Returns:
            Dictionary mapping report types to file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_reports = {}
        
        print("📄 Generating standardized reports...")
        
        # Generate individual reports
        for result in results:
            try:
                # Extract accuracy metrics for this specific result if available
                accuracy_metrics = None
                if accuracy_report and "ground_truth_evaluation" in accuracy_report:
                    gt_eval = accuracy_report["ground_truth_evaluation"]
                    if result.implementation_name in gt_eval and "error" not in gt_eval[result.implementation_name]:
                        from standardized_data_models import AccuracyMetrics
                        metrics_data = gt_eval[result.implementation_name]
                        accuracy_metrics = AccuracyMetrics(
                            precision=metrics_data["precision"],
                            recall=metrics_data["recall"],
                            f1_score=metrics_data["f1_score"],
                            iou=metrics_data["iou"],
                            accuracy=metrics_data["accuracy"],
                            specificity=metrics_data["specificity"]
                        )
                
                # Generate individual report
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                individual_path = os.path.join(output_dir, f"{result.implementation_name.lower().replace(' ', '_')}_report_{timestamp}.pdf")
                report_path = self.report_generator.generate_individual_report(
                    result, accuracy_metrics, individual_path
                )
                generated_reports[f"{result.implementation_name}_individual"] = report_path
                
            except Exception as e:
                print(f"❌ Failed to generate individual report for {result.implementation_name}: {e}")
        
        # Generate comparison report if multiple results
        if len(results) > 1:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                comparison_path = os.path.join(output_dir, f"comparison_report_{timestamp}.pdf")
                report_path = self.report_generator.generate_comparison_report(
                    results, accuracy_report, comparison_path
                )
                generated_reports["comparison"] = report_path
            except Exception as e:
                print(f"❌ Failed to generate comparison report: {e}")
        
        # Generate executive summary
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            executive_path = os.path.join(output_dir, f"executive_summary_{timestamp}.pdf")
            report_path = self.report_generator.generate_executive_summary_report(
                results, accuracy_report, executive_path
            )
            generated_reports["executive_summary"] = report_path
        except Exception as e:
            print(f"❌ Failed to generate executive summary: {e}")
        
        if generated_reports:
            print(f"✅ Generated {len(generated_reports)} report(s) in {output_dir}/")
        else:
            print("❌ No reports were generated successfully")
        
        return generated_reports
    
    def print_comparison(self, comparison: Dict[str, Any]):
        """Print a formatted comparison of results."""
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        print(f"Number of implementations: {comparison['num_implementations']}")
        print(f"Agreement level: {comparison['summary']['agreement_level']}")
        print(f"Fastest implementation: {comparison['summary']['fastest_implementation']}")
        print(f"Most changes detected: {comparison['summary']['most_changes_detected']}")
        print(f"Most regions detected: {comparison['summary']['most_regions_detected']}")
        
        print(f"\nProcessing Times:")
        for name, time in comparison['processing_times'].items():
            print(f"  {name}: {time:.3f} seconds")
        
        print(f"\nChange Detection Metrics:")
        metrics = comparison['metrics_comparison']
        print(f"  Change Pixels - Min: {metrics['change_pixels']['minimum']:,}, "
              f"Max: {metrics['change_pixels']['maximum']:,}, "
              f"Avg: {metrics['change_pixels']['average']:,.0f}")
        print(f"  Regions - Min: {metrics['num_regions']['minimum']}, "
              f"Max: {metrics['num_regions']['maximum']}, "
              f"Avg: {metrics['num_regions']['average']:.1f}")
        print(f"  Confidence - Min: {metrics['confidence']['minimum']:.3f}, "
              f"Max: {metrics['confidence']['maximum']:.3f}, "
              f"Avg: {metrics['confidence']['average']:.3f}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Run change detection implementations with accuracy evaluation")
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument("--implementation", "-i", 
                       choices=['basic', 'advanced', 'deep_learning', 'all'],
                       default='all',
                       help="Implementation to run (default: all)")
    parser.add_argument("--output-dir", "-o", default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum area threshold for change regions (default: 100)")
    parser.add_argument("--ground-truth", "-g", 
                       help="Path to ground truth mask for accuracy evaluation")
    parser.add_argument("--save", action="store_true",
                       help="Save results to files")
    parser.add_argument("--evaluate", action="store_true",
                       help="Perform accuracy evaluation (automatically enabled with ground truth)")
    parser.add_argument("--generate-reports", action="store_true",
                       help="Generate standardized PDF reports")
    parser.add_argument("--reports-dir", default="reports",
                       help="Directory for generated reports (default: reports)")
    
    args = parser.parse_args()
    
    # Check input files
    if not os.path.exists(args.img1):
        print(f"Error: Image file {args.img1} not found")
        return
    
    if not os.path.exists(args.img2):
        print(f"Error: Image file {args.img2} not found")
        return
    
    # Check ground truth file if provided
    if args.ground_truth and not os.path.exists(args.ground_truth):
        print(f"Warning: Ground truth file {args.ground_truth} not found")
        args.ground_truth = None
    
    # Create runner
    runner = ChangeDetectionRunner()
    
    # Run implementations
    if args.implementation == 'all':
        results = runner.run_all_implementations(args.img1, args.img2, min_area=args.min_area)
    else:
        result = runner.run_implementation(args.implementation, args.img1, args.img2, min_area=args.min_area)
        results = [result]
    
    if not results:
        print("No results to process")
        return
    
    # Compare results if multiple implementations
    comparison = {}
    if len(results) > 1:
        comparison = runner.compare_results(results)
        runner.print_comparison(comparison)
    
    # Perform accuracy evaluation
    accuracy_report = None
    if args.evaluate or args.ground_truth or len(results) > 1:
        print("\n🔍 Performing accuracy evaluation...")
        accuracy_report = runner.evaluate_accuracy(results, args.ground_truth)
        runner.evaluator.print_accuracy_summary(accuracy_report)
    
    # Generate reports if requested
    if args.generate_reports:
        generated_reports = runner.generate_reports(results, accuracy_report, args.reports_dir)
        if generated_reports:
            print("\n📄 Generated Reports:")
            for report_type, path in generated_reports.items():
                print(f"  • {report_type}: {path}")
    
    # Save results if requested
    if args.save:
        runner.save_results(results, comparison, accuracy_report, args.output_dir)
    
    print(f"\n🎉 Processing complete! Analyzed {len(results)} implementation(s)")
    if accuracy_report:
        print("📊 Accuracy evaluation completed and included in results")
    if args.generate_reports and generated_reports:
        print(f"📄 Generated {len(generated_reports)} standardized report(s)")


if __name__ == "__main__":
    main()