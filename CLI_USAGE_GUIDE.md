# Change Detection System - CLI Usage Guide

This guide covers the new command-line interface and configuration management system for the Change Detection System.

## Overview

The CLI system provides a comprehensive interface for running change detection implementations with:
- Configuration file management
- Flexible command-line options
- Comprehensive logging and progress tracking
- Validation and error handling
- Integration with all existing functionality

## Quick Start

### Basic Usage

```bash
# Run all implementations with default settings
python cli.py image1.jpg image2.jpg

# Run specific implementation
python cli.py image1.jpg image2.jpg -i basic

# Run with reports and accuracy evaluation
python cli.py image1.jpg image2.jpg --generate-reports --save
```

### With Ground Truth

```bash
# Run with ground truth for accuracy evaluation
python cli.py image1.jpg image2.jpg -g ground_truth.png --generate-reports
```

## Configuration Management

### Configuration File

The system uses a YAML configuration file (`config.yaml`) to set default parameters:

```yaml
# Input Configuration
input:
  image1_path: ""
  image2_path: ""
  ground_truth_path: ""

# Processing Configuration
processing:
  implementations: ['basic', 'advanced', 'deep_learning']
  min_area_threshold: 100
  confidence_threshold: 0.5

# Implementation-specific parameters
implementations:
  basic:
    blur_kernel_size: 5
    threshold_value: 30
  advanced:
    gaussian_blur_sigma: 1.0
    edge_threshold_low: 50
  deep_learning:
    patch_size: 32
    similarity_threshold: 0.7

# Output Configuration
output:
  base_directory: "results"
  include_visualizations: true

# Logging Configuration
logging:
  level: "INFO"
  log_to_file: true
  show_progress: true
```

### Configuration Commands

```bash
# Show current configuration
python cli.py --show-config

# Validate configuration
python cli.py --validate-config

# Use custom configuration file
python cli.py image1.jpg image2.jpg --config my_config.yaml

# Save current configuration
python cli.py --save-config my_saved_config.yaml
```

## Command-Line Options

### Implementation Selection

```bash
# Run specific implementation
python cli.py image1.jpg image2.jpg -i basic

# Run multiple specific implementations
python cli.py image1.jpg image2.jpg -i advanced -i deep_learning

# Run all implementations (default)
python cli.py image1.jpg image2.jpg -i all
```

### Input/Output Options

```bash
# Custom output directory
python cli.py image1.jpg image2.jpg -o my_results

# With ground truth
python cli.py image1.jpg image2.jpg -g ground_truth.png

# Custom reports directory
python cli.py image1.jpg image2.jpg --generate-reports --reports-dir my_reports
```

### Processing Parameters

```bash
# Custom minimum area threshold
python cli.py image1.jpg image2.jpg --min-area 200

# Custom confidence threshold
python cli.py image1.jpg image2.jpg --confidence-threshold 0.7
```

### Feature Control

```bash
# Save results to files
python cli.py image1.jpg image2.jpg --save

# Generate reports
python cli.py image1.jpg image2.jpg --generate-reports

# Perform accuracy evaluation
python cli.py image1.jpg image2.jpg --evaluate

# Disable reports
python cli.py image1.jpg image2.jpg --no-reports

# Disable accuracy evaluation
python cli.py image1.jpg image2.jpg --no-accuracy
```

### Logging and Output Control

```bash
# Verbose logging (DEBUG level)
python cli.py image1.jpg image2.jpg --verbose

# Quiet mode (WARNING level and above)
python cli.py image1.jpg image2.jpg --quiet

# Disable progress bars
python cli.py image1.jpg image2.jpg --no-progress

# Custom log file
python cli.py image1.jpg image2.jpg --log-file my_log.txt
```

### Utility Commands

```bash
# List available implementations
python cli.py --list-implementations

# Show current configuration
python cli.py --show-config

# Validate configuration
python cli.py --validate-config

# Show help
python cli.py --help
```

## Advanced Usage Examples

### Complete Workflow

```bash
# Run complete analysis with all features
python cli.py orlando2010.png orlando2023.png \
  --generate-reports \
  --save \
  --verbose \
  --output-dir orlando_analysis \
  --reports-dir orlando_reports
```

### Custom Configuration

```bash
# Create custom configuration
python cli.py --save-config custom_config.yaml

# Edit custom_config.yaml as needed

# Run with custom configuration
python cli.py image1.jpg image2.jpg --config custom_config.yaml
```

### Specific Implementation Analysis

```bash
# Compare only advanced methods
python cli.py image1.jpg image2.jpg \
  -i advanced \
  -i deep_learning \
  --generate-reports \
  --min-area 150 \
  --confidence-threshold 0.6
```

### Batch Processing Setup

```bash
# Set up for batch processing
python cli.py --save-config batch_config.yaml

# Edit batch_config.yaml to set:
# - processing.parallel_execution: true
# - logging.level: "WARNING"
# - output.save_intermediate: false

# Run batch processing
python cli.py image1.jpg image2.jpg --config batch_config.yaml --quiet
```

## Logging and Monitoring

### Log Levels

- `DEBUG`: Detailed information for debugging
- `INFO`: General information (default)
- `WARNING`: Warning messages only
- `ERROR`: Error messages only
- `CRITICAL`: Critical errors only

### Log Output

The system provides:
- Console output with timestamps
- File logging (optional)
- Progress bars for long operations
- Performance monitoring and metrics
- System information logging

### Log Files

By default, logs are saved to `results/change_detection.log`. You can customize:

```bash
# Custom log file location
python cli.py image1.jpg image2.jpg --log-file /path/to/my.log

# Disable file logging (console only)
# Edit config.yaml: logging.log_to_file: false
```

## Error Handling

The CLI system provides comprehensive error handling:

### Configuration Errors

```bash
# Validate configuration before running
python cli.py --validate-config
```

Common configuration errors:
- Missing or invalid image paths
- Invalid implementation names
- Out-of-range parameter values
- Invalid log levels

### Runtime Errors

The system handles:
- Missing input files
- Implementation failures
- Memory/resource constraints
- Output directory creation issues

### Recovery Options

```bash
# Continue with available implementations if some fail
python cli.py image1.jpg image2.jpg --verbose

# Use minimal configuration for troubleshooting
python cli.py image1.jpg image2.jpg -i basic --no-reports --quiet
```

## Integration with Existing Code

The new CLI system is fully compatible with existing code:

### Direct Usage

```python
from cli import main
import sys

# Set command-line arguments
sys.argv = ['cli.py', 'image1.jpg', 'image2.jpg', '--save']
main()
```

### Configuration in Code

```python
from config_manager import ConfigManager
from logging_utils import setup_logging

# Load configuration
config_manager = ConfigManager()
config = config_manager.load_config()

# Set up logging
logger, progress_tracker, performance_monitor = setup_logging(config)

# Use in your code
logger.info("Starting processing...")
```

## Migration from Old System

### Old unified_runner.py

```bash
# Old way
python unified_runner.py image1.jpg image2.jpg --implementation all --save

# New way
python cli.py image1.jpg image2.jpg --save
```

### Benefits of New System

1. **Configuration Management**: Centralized settings in YAML files
2. **Better Logging**: Structured logging with multiple levels
3. **Progress Tracking**: Visual progress bars and performance monitoring
4. **Validation**: Input validation and error checking
5. **Flexibility**: More command-line options and combinations
6. **Documentation**: Comprehensive help and examples

## Troubleshooting

### Common Issues

1. **Configuration file not found**
   ```bash
   python cli.py --show-config  # Check current config
   ```

2. **Invalid image paths**
   ```bash
   python cli.py --validate-config  # Validate before running
   ```

3. **Permission errors**
   ```bash
   python cli.py image1.jpg image2.jpg -o /tmp/results  # Use different output dir
   ```

4. **Memory issues**
   ```bash
   python cli.py image1.jpg image2.jpg -i basic --no-progress  # Use minimal resources
   ```

### Getting Help

```bash
# Show all available options
python cli.py --help

# List implementations
python cli.py --list-implementations

# Show current configuration
python cli.py --show-config

# Validate setup
python cli.py --validate-config
```

## Performance Tips

1. **Use specific implementations** instead of running all for faster processing
2. **Disable progress bars** (`--no-progress`) for batch processing
3. **Use WARNING log level** (`--quiet`) to reduce output overhead
4. **Disable intermediate saves** in config for faster processing
5. **Use appropriate minimum area thresholds** to filter noise

## Future Enhancements

The CLI system is designed to be extensible:
- Parallel processing support
- Additional output formats
- Plugin system for custom implementations
- Web interface integration
- Batch processing utilities