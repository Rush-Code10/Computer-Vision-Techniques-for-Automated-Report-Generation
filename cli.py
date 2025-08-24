#!/usr/bin/env python3
"""
Command-line interface for the Change Detection System.
Provides a comprehensive CLI with configuration management, logging, and workflow control.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

from config_manager import ConfigManager, SystemConfig
from logging_utils import setup_logging
from unified_runner import ChangeDetectionRunner


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Change Detection System - Analyze infrastructure development from satellite imagery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all implementations with default settings
  python cli.py image1.jpg image2.jpg
  
  # Run specific implementation with custom output directory
  python cli.py image1.jpg image2.jpg -i basic -o my_results
  
  # Run with ground truth for accuracy evaluation
  python cli.py image1.jpg image2.jpg -g ground_truth.png --generate-reports
  
  # Use custom configuration file
  python cli.py image1.jpg image2.jpg --config my_config.yaml
  
  # Run with verbose logging and progress tracking
  python cli.py image1.jpg image2.jpg --verbose --save --generate-reports
  
  # Run specific implementations only
  python cli.py image1.jpg image2.jpg -i advanced -i deep_learning
        """
    )
    
    # Required arguments (but not for utility commands)
    parser.add_argument("img1", nargs='?', help="Path to first (earlier) satellite image")
    parser.add_argument("img2", nargs='?', help="Path to second (later) satellite image")
    
    # Implementation selection
    impl_group = parser.add_argument_group("Implementation Selection")
    impl_group.add_argument(
        "--implementation", "-i",
        action="append",
        choices=['basic', 'advanced', 'deep_learning', 'all'],
        help="Implementation(s) to run. Can be specified multiple times. Use 'all' for all implementations."
    )
    
    # Input/Output options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--ground-truth", "-g",
        help="Path to ground truth mask for accuracy evaluation"
    )
    io_group.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Output directory for results (default: results)"
    )
    io_group.add_argument(
        "--config", "-c",
        help="Path to configuration file (default: config.yaml)"
    )
    io_group.add_argument(
        "--save-config",
        help="Save current configuration to specified file"
    )
    
    # Processing parameters
    proc_group = parser.add_argument_group("Processing Parameters")
    proc_group.add_argument(
        "--min-area",
        type=int,
        help="Minimum area threshold for change regions (pixels)"
    )
    proc_group.add_argument(
        "--confidence-threshold",
        type=float,
        help="Confidence threshold for filtering results (0.0-1.0)"
    )
    
    # Feature flags
    feature_group = parser.add_argument_group("Feature Control")
    feature_group.add_argument(
        "--save",
        action="store_true",
        help="Save results to files"
    )
    feature_group.add_argument(
        "--generate-reports",
        action="store_true",
        help="Generate standardized PDF reports"
    )
    feature_group.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for generated reports (default: reports)"
    )
    feature_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Perform accuracy evaluation (automatically enabled with ground truth)"
    )
    feature_group.add_argument(
        "--no-reports",
        action="store_true",
        help="Disable report generation"
    )
    feature_group.add_argument(
        "--no-accuracy",
        action="store_true",
        help="Disable accuracy evaluation"
    )
    
    # Logging and output control
    log_group = parser.add_argument_group("Logging and Output")
    log_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging"
    )
    log_group.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output (WARNING level and above)"
    )
    log_group.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    log_group.add_argument(
        "--log-file",
        help="Custom log file path"
    )
    
    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument(
        "--list-implementations",
        action="store_true",
        help="List available implementations and exit"
    )
    util_group.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    util_group.add_argument(
        "--show-config",
        action="store_true",
        help="Show current configuration and exit"
    )
    
    return parser


def handle_utility_commands(args: argparse.Namespace) -> bool:
    """
    Handle utility commands that don't require full processing.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if a utility command was handled (should exit), False otherwise
    """
    if args.list_implementations:
        print("Available Change Detection Implementations:")
        print("  • basic          - Basic computer vision approach")
        print("  • advanced       - Advanced computer vision with edge detection")
        print("  • deep_learning  - Deep learning inspired approach")
        print("  • all            - Run all implementations")
        return True
    
    if args.validate_config or args.show_config:
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
        config = config_manager.merge_with_args(args)
        
        if args.validate_config:
            errors = config_manager.validate_config()
            if errors:
                print("Configuration Validation Errors:")
                for error in errors:
                    print(f"  ❌ {error}")
                return True
            else:
                print("✅ Configuration is valid")
                return True
        
        if args.show_config:
            config_manager.print_config_summary()
            return True
    
    return False


def process_implementation_args(args: argparse.Namespace) -> List[str]:
    """
    Process implementation arguments and return list of implementations to run.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of implementation names
    """
    if not args.implementation:
        return ['basic', 'advanced', 'deep_learning']  # Default to all
    
    implementations = []
    for impl in args.implementation:
        if impl == 'all':
            return ['basic', 'advanced', 'deep_learning']
        else:
            implementations.append(impl)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(implementations))


def main():
    """Main CLI function."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Handle utility commands
    if handle_utility_commands(args):
        return 0
    
    # Check required arguments for non-utility commands
    if not args.img1 or not args.img2:
        parser.error("img1 and img2 are required for processing commands")
    
    # Process implementation arguments
    if args.implementation:
        args.implementation = process_implementation_args(args)
    
    # Load and merge configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.load_config()
    config = config_manager.merge_with_args(args)
    
    # Validate configuration
    errors = config_manager.validate_config()
    if errors:
        print("Configuration Errors:")
        for error in errors:
            print(f"  ❌ {error}")
        return 1
    
    # Save configuration if requested
    if args.save_config:
        config_manager.save_config(args.save_config)
        print(f"Configuration saved to {args.save_config}")
        return 0
    
    # Set up logging and monitoring
    logger, progress_tracker, performance_monitor = setup_logging(config)
    
    try:
        # Log system information and configuration
        logger.log_system_info()
        
        # Create output directories
        os.makedirs(config.base_directory, exist_ok=True)
        if args.generate_reports and not args.no_reports:
            os.makedirs(args.reports_dir, exist_ok=True)
        
        # Start performance monitoring
        performance_monitor.start_monitoring()
        
        # Create runner
        runner = ChangeDetectionRunner()
        
        # Run implementations
        with logger.step_context("Running implementations"):
            if len(config.implementations) == 1:
                logger.info(f"Running {config.implementations[0]} implementation...")
                result = runner.run_implementation(
                    config.implementations[0],
                    config.image1_path,
                    config.image2_path,
                    min_area=config.min_area_threshold,
                    **config_manager.get_implementation_params(config.implementations[0])
                )
                results = [result]
            else:
                logger.info(f"Running {len(config.implementations)} implementations...")
                results = []
                
                with progress_tracker.progress_context(
                    "implementations", 
                    len(config.implementations), 
                    "Processing implementations"
                ) as pbar:
                    for impl_name in config.implementations:
                        try:
                            performance_monitor.checkpoint(f"start_{impl_name}")
                            
                            result = runner.run_implementation(
                                impl_name,
                                config.image1_path,
                                config.image2_path,
                                min_area=config.min_area_threshold,
                                **config_manager.get_implementation_params(impl_name)
                            )
                            results.append(result)
                            
                            performance_monitor.checkpoint(f"end_{impl_name}")
                            if pbar:
                                pbar.update(1)
                                
                        except Exception as e:
                            logger.error(f"Failed to run {impl_name}: {e}")
                            if pbar:
                                pbar.update(1)
        
        if not results:
            logger.error("No implementations completed successfully")
            return 1
        
        # Log results summary
        logger.log_results_summary(results)
        
        # Compare results if multiple implementations
        comparison = {}
        if len(results) > 1:
            with logger.step_context("Comparing results"):
                comparison = runner.compare_results(results)
                runner.print_comparison(comparison)
        
        # Perform accuracy evaluation
        accuracy_report = None
        if (config.calculate_accuracy or config.ground_truth_path or 
            args.evaluate) and not args.no_accuracy:
            with logger.step_context("Accuracy evaluation"):
                accuracy_report = runner.evaluate_accuracy(results, config.ground_truth_path)
                runner.evaluator.print_accuracy_summary(accuracy_report)
        
        # Generate reports
        generated_reports = {}
        if args.generate_reports and not args.no_reports:
            with logger.step_context("Generating reports"):
                generated_reports = runner.generate_reports(
                    results, accuracy_report, args.reports_dir
                )
                if generated_reports:
                    logger.info("Generated Reports:")
                    for report_type, path in generated_reports.items():
                        logger.info(f"  • {report_type}: {path}")
        
        # Save results
        if args.save:
            with logger.step_context("Saving results"):
                runner.save_results(results, comparison, accuracy_report, config.base_directory)
        
        # Log performance summary
        performance_monitor.log_summary()
        logger.log_performance_metrics()
        
        logger.info(f"🎉 Processing complete! Analyzed {len(results)} implementation(s)")
        if accuracy_report:
            logger.info("📊 Accuracy evaluation completed")
        if generated_reports:
            logger.info(f"📄 Generated {len(generated_reports)} report(s)")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.debug("Full traceback:", exc_info=True)
        return 1
    finally:
        # Clean up progress tracking
        progress_tracker.close_all()


if __name__ == "__main__":
    sys.exit(main())