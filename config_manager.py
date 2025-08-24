"""
Configuration management for the Change Detection System.
Handles loading, validation, and merging of configuration files with command-line arguments.
"""

import os
import yaml
import argparse
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class SystemConfig:
    """System configuration data structure."""
    # Input settings
    image1_path: str = ""
    image2_path: str = ""
    ground_truth_path: str = ""
    
    # Processing settings
    implementations: List[str] = None
    parallel_execution: bool = False
    min_area_threshold: int = 100
    confidence_threshold: float = 0.5
    
    # Implementation parameters
    implementation_params: Dict[str, Dict[str, Any]] = None
    
    # Output settings
    base_directory: str = "results"
    report_formats: List[str] = None
    include_visualizations: bool = True
    save_intermediate: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    
    # Evaluation settings
    calculate_accuracy: bool = True
    calculate_agreement: bool = True
    generate_accuracy_plots: bool = True
    min_region_size: int = 10
    
    # Logging settings
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "change_detection.log"
    include_timestamps: bool = True
    show_progress: bool = True
    
    # Report settings
    include_executive_summary: bool = True
    include_comparison: bool = True
    include_accuracy_analysis: bool = True
    template_style: str = "standard"
    include_metadata: bool = True
    
    def __post_init__(self):
        """Initialize default values."""
        if self.implementations is None:
            self.implementations = ['basic', 'advanced', 'deep_learning']
        if self.report_formats is None:
            self.report_formats = ['pdf']
        if self.implementation_params is None:
            self.implementation_params = {}


class ConfigManager:
    """Manages configuration loading and validation."""
    
    DEFAULT_CONFIG_FILE = "config.yaml"
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config = SystemConfig()
    
    def load_config(self, config_file: Optional[str] = None) -> SystemConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            SystemConfig object with loaded settings
        """
        if config_file:
            self.config_file = config_file
        
        if not os.path.exists(self.config_file):
            print(f"Warning: Configuration file {self.config_file} not found. Using defaults.")
            return self.config
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                self._update_config_from_dict(config_data)
                print(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            print(f"Error loading configuration file {self.config_file}: {e}")
            print("Using default configuration.")
        
        return self.config
    
    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        # Input settings
        if 'input' in config_data:
            input_config = config_data['input']
            self.config.image1_path = input_config.get('image1_path', self.config.image1_path)
            self.config.image2_path = input_config.get('image2_path', self.config.image2_path)
            self.config.ground_truth_path = input_config.get('ground_truth_path', self.config.ground_truth_path)
        
        # Processing settings
        if 'processing' in config_data:
            proc_config = config_data['processing']
            implementations = proc_config.get('implementations', self.config.implementations)
            if implementations == 'all':
                self.config.implementations = ['basic', 'advanced', 'deep_learning']
            elif isinstance(implementations, list):
                self.config.implementations = implementations
            
            self.config.parallel_execution = proc_config.get('parallel_execution', self.config.parallel_execution)
            self.config.min_area_threshold = proc_config.get('min_area_threshold', self.config.min_area_threshold)
            self.config.confidence_threshold = proc_config.get('confidence_threshold', self.config.confidence_threshold)
        
        # Implementation parameters
        if 'implementations' in config_data:
            self.config.implementation_params = config_data['implementations']
        
        # Output settings
        if 'output' in config_data:
            output_config = config_data['output']
            self.config.base_directory = output_config.get('base_directory', self.config.base_directory)
            self.config.report_formats = output_config.get('report_formats', self.config.report_formats)
            self.config.include_visualizations = output_config.get('include_visualizations', self.config.include_visualizations)
            self.config.save_intermediate = output_config.get('save_intermediate', self.config.save_intermediate)
            self.config.timestamp_format = output_config.get('timestamp_format', self.config.timestamp_format)
        
        # Evaluation settings
        if 'evaluation' in config_data:
            eval_config = config_data['evaluation']
            self.config.calculate_accuracy = eval_config.get('calculate_accuracy', self.config.calculate_accuracy)
            self.config.calculate_agreement = eval_config.get('calculate_agreement', self.config.calculate_agreement)
            self.config.generate_accuracy_plots = eval_config.get('generate_accuracy_plots', self.config.generate_accuracy_plots)
            self.config.min_region_size = eval_config.get('min_region_size', self.config.min_region_size)
        
        # Logging settings
        if 'logging' in config_data:
            log_config = config_data['logging']
            self.config.log_level = log_config.get('level', self.config.log_level)
            self.config.log_to_file = log_config.get('log_to_file', self.config.log_to_file)
            self.config.log_file = log_config.get('log_file', self.config.log_file)
            self.config.include_timestamps = log_config.get('include_timestamps', self.config.include_timestamps)
            self.config.show_progress = log_config.get('show_progress', self.config.show_progress)
        
        # Report settings
        if 'reports' in config_data:
            report_config = config_data['reports']
            self.config.include_executive_summary = report_config.get('include_executive_summary', self.config.include_executive_summary)
            self.config.include_comparison = report_config.get('include_comparison', self.config.include_comparison)
            self.config.include_accuracy_analysis = report_config.get('include_accuracy_analysis', self.config.include_accuracy_analysis)
            self.config.template_style = report_config.get('template_style', self.config.template_style)
            self.config.include_metadata = report_config.get('include_metadata', self.config.include_metadata)
    
    def merge_with_args(self, args: argparse.Namespace) -> SystemConfig:
        """
        Merge configuration with command-line arguments.
        Command-line arguments take precedence over config file.
        
        Args:
            args: Parsed command-line arguments
            
        Returns:
            Updated SystemConfig object
        """
        # Override with command-line arguments
        if hasattr(args, 'img1') and args.img1:
            self.config.image1_path = args.img1
        if hasattr(args, 'img2') and args.img2:
            self.config.image2_path = args.img2
        if hasattr(args, 'ground_truth') and args.ground_truth:
            self.config.ground_truth_path = args.ground_truth
        
        if hasattr(args, 'implementation') and args.implementation:
            # args.implementation is now a list after processing
            if isinstance(args.implementation, list):
                self.config.implementations = args.implementation
            elif args.implementation == 'all':
                self.config.implementations = ['basic', 'advanced', 'deep_learning']
            else:
                self.config.implementations = [args.implementation]
        
        if hasattr(args, 'output_dir') and args.output_dir:
            self.config.base_directory = args.output_dir
        if hasattr(args, 'min_area') and args.min_area is not None:
            self.config.min_area_threshold = args.min_area
        if hasattr(args, 'confidence_threshold') and args.confidence_threshold is not None:
            self.config.confidence_threshold = args.confidence_threshold
        
        # Logging settings
        if hasattr(args, 'verbose') and args.verbose:
            self.config.log_level = "DEBUG"
        if hasattr(args, 'quiet') and args.quiet:
            self.config.log_level = "WARNING"
        if hasattr(args, 'no_progress') and args.no_progress:
            self.config.show_progress = False
        
        # Feature flags
        if hasattr(args, 'no_reports') and args.no_reports:
            self.config.include_executive_summary = False
            self.config.include_comparison = False
        if hasattr(args, 'no_accuracy') and args.no_accuracy:
            self.config.calculate_accuracy = False
            self.config.calculate_agreement = False
        
        return self.config
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate required paths
        if not self.config.image1_path:
            errors.append("image1_path is required")
        elif not os.path.exists(self.config.image1_path):
            errors.append(f"Image 1 file not found: {self.config.image1_path}")
        
        if not self.config.image2_path:
            errors.append("image2_path is required")
        elif not os.path.exists(self.config.image2_path):
            errors.append(f"Image 2 file not found: {self.config.image2_path}")
        
        if self.config.ground_truth_path and not os.path.exists(self.config.ground_truth_path):
            errors.append(f"Ground truth file not found: {self.config.ground_truth_path}")
        
        # Validate implementations
        valid_implementations = ['basic', 'advanced', 'deep_learning']
        for impl in self.config.implementations:
            if impl not in valid_implementations:
                errors.append(f"Invalid implementation: {impl}. Valid options: {valid_implementations}")
        
        # Validate numeric parameters
        if self.config.min_area_threshold < 0:
            errors.append("min_area_threshold must be non-negative")
        
        if not 0 <= self.config.confidence_threshold <= 1:
            errors.append("confidence_threshold must be between 0 and 1")
        
        if self.config.min_region_size < 0:
            errors.append("min_region_size must be non-negative")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.config.log_level not in valid_log_levels:
            errors.append(f"Invalid log level: {self.config.log_level}. Valid options: {valid_log_levels}")
        
        return errors
    
    def get_implementation_params(self, implementation_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific implementation.
        
        Args:
            implementation_name: Name of the implementation
            
        Returns:
            Dictionary of parameters for the implementation
        """
        return self.config.implementation_params.get(implementation_name, {})
    
    def save_config(self, output_path: str):
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path to save the configuration file
        """
        config_dict = {
            'input': {
                'image1_path': self.config.image1_path,
                'image2_path': self.config.image2_path,
                'ground_truth_path': self.config.ground_truth_path
            },
            'processing': {
                'implementations': self.config.implementations,
                'parallel_execution': self.config.parallel_execution,
                'min_area_threshold': self.config.min_area_threshold,
                'confidence_threshold': self.config.confidence_threshold
            },
            'implementations': self.config.implementation_params,
            'output': {
                'base_directory': self.config.base_directory,
                'report_formats': self.config.report_formats,
                'include_visualizations': self.config.include_visualizations,
                'save_intermediate': self.config.save_intermediate,
                'timestamp_format': self.config.timestamp_format
            },
            'evaluation': {
                'calculate_accuracy': self.config.calculate_accuracy,
                'calculate_agreement': self.config.calculate_agreement,
                'generate_accuracy_plots': self.config.generate_accuracy_plots,
                'min_region_size': self.config.min_region_size
            },
            'logging': {
                'level': self.config.log_level,
                'log_to_file': self.config.log_to_file,
                'log_file': self.config.log_file,
                'include_timestamps': self.config.include_timestamps,
                'show_progress': self.config.show_progress
            },
            'reports': {
                'include_executive_summary': self.config.include_executive_summary,
                'include_comparison': self.config.include_comparison,
                'include_accuracy_analysis': self.config.include_accuracy_analysis,
                'template_style': self.config.template_style,
                'include_metadata': self.config.include_metadata
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {output_path}")
        except Exception as e:
            print(f"Error saving configuration to {output_path}: {e}")
    
    def print_config_summary(self):
        """Print a summary of the current configuration."""
        print("\n" + "=" * 50)
        print("CONFIGURATION SUMMARY")
        print("=" * 50)
        print(f"Input Images: {self.config.image1_path} -> {self.config.image2_path}")
        if self.config.ground_truth_path:
            print(f"Ground Truth: {self.config.ground_truth_path}")
        print(f"Implementations: {', '.join(self.config.implementations)}")
        print(f"Output Directory: {self.config.base_directory}")
        print(f"Min Area Threshold: {self.config.min_area_threshold} pixels")
        print(f"Confidence Threshold: {self.config.confidence_threshold}")
        print(f"Log Level: {self.config.log_level}")
        print(f"Generate Reports: {self.config.include_executive_summary or self.config.include_comparison}")
        print(f"Calculate Accuracy: {self.config.calculate_accuracy}")
        print("=" * 50)