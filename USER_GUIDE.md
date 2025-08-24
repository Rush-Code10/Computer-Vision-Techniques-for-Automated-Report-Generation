# Change Detection System - User Guide

## Overview

The Change Detection System is a comprehensive tool for analyzing infrastructure development from satellite imagery. It provides three different change detection implementations with standardized reporting, accuracy evaluation, and visualization capabilities.

## Quick Start

### Basic Usage

```bash
# Run all implementations with Orlando airport images
python cli.py orlando2010.png orlando2023.png

# Run with reports and save results
python cli.py orlando2010.png orlando2023.png --generate-reports --save
python cli.py lv2010.png lv2022.png --generate-reports --save
```

### System Requirements

- Python 3.7+
- Required packages: numpy, opencv-python, matplotlib, scikit-learn, reportlab, PyYAML
- Input: Two satellite images (PNG, JPG, or other OpenCV-supported formats)
- Output: Results, reports, and visualizations

## System Components

### 1. Change Detection Implementations

The system includes three different approaches:

#### Basic Computer Vision
- Simple difference-based approach
- Fast processing
- Good for obvious changes
- Uses thresholding and morphological operations

#### Advanced Computer Vision  
- Edge detection and feature analysis
- More sophisticated filtering
- Better noise reduction
- Handles complex scenarios

#### Deep Learning Inspired
- Patch-based similarity analysis
- Confidence scoring
- Handles subtle changes
- More computationally intensive

### 2. Command-Line Interface (CLI)

The CLI provides comprehensive control over the system:

```bash
python cli.py [options] image1 image2
```

Key features:
- Configuration management
- Implementation selection
- Report generation
- Accuracy evaluation
- Logging and progress tracking

### 3. Configuration System

Uses YAML configuration files for settings:

```yaml
# config.yaml
input:
  image1_path: ""
  image2_path: ""
  ground_truth_path: ""

processing:
  implementations: ['basic', 'advanced', 'deep_learning']
  min_area_threshold: 100
  confidence_threshold: 0.5

output:
  base_directory: "results"
  include_visualizations: true
```

## Detailed Usage

### Running Individual Implementations

```bash
# Run only basic implementation
python cli.py image1.jpg image2.jpg -i basic

# Run advanced and deep learning only
python cli.py image1.jpg image2.jpg -i advanced -i deep_learning
```

### Output Management

```bash
# Custom output directory
python cli.py image1.jpg image2.jpg -o my_results

# Custom reports directory
python cli.py image1.jpg image2.jpg --generate-reports --reports-dir my_reports

# Save all results
python cli.py image1.jpg image2.jpg --save
```

### Accuracy Evaluation

```bash
# With ground truth mask
python cli.py image1.jpg image2.jpg -g ground_truth.png

# Force accuracy evaluation (inter-method comparison)
python cli.py image1.jpg image2.jpg --evaluate
```

### Processing Parameters

```bash
# Custom minimum area threshold (pixels)
python cli.py image1.jpg image2.jpg --min-area 200

# Custom confidence threshold
python cli.py image1.jpg image2.jpg --confidence-threshold 0.7
```

### Logging and Monitoring

```bash
# Verbose logging
python cli.py image1.jpg image2.jpg --verbose

# Quiet mode (warnings only)
python cli.py image1.jpg image2.jpg --quiet

# Custom log file
python cli.py image1.jpg image2.jpg --log-file my_analysis.log
```

## Understanding Results

### Output Files

The system generates several types of output files:

#### Results Directory
```
results/
├── comparison_summary.json      # Comparison between implementations
├── accuracy_evaluation.json     # Accuracy metrics and evaluation
├── basic_computer_vision_result.json    # Individual results
├── basic_computer_vision_mask.npy       # Change masks
├── advanced_computer_vision_result.json
├── advanced_computer_vision_mask.npy
├── deep_learning_inspired_result.json
├── deep_learning_inspired_mask.npy
└── deep_learning_inspired_confidence.npy
```

#### Reports Directory
```
reports/
├── basic_computer_vision_report_TIMESTAMP.pdf
├── advanced_computer_vision_report_TIMESTAMP.pdf
├── deep_learning_inspired_report_TIMESTAMP.pdf
├── comparison_report_TIMESTAMP.pdf
└── executive_summary_TIMESTAMP.pdf
```

### Result Interpretation

#### Change Detection Metrics
- **Total Change Area**: Area of detected changes in pixels
- **Number of Regions**: Count of distinct change regions
- **Average Confidence**: Mean confidence score (0-1)
- **Processing Time**: Time taken for analysis

#### Accuracy Metrics (with ground truth)
- **Precision**: Accuracy of positive predictions
- **Recall**: Coverage of actual changes
- **F1-Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union score
- **Accuracy**: Overall correctness

#### Inter-Method Agreement
When no ground truth is available, the system compares methods:
- **High Agreement**: Methods detect similar changes
- **Medium Agreement**: Some differences between methods
- **Low Agreement**: Significant differences between methods

## Report Types

### Individual Reports
- Method description and parameters
- Change detection results
- Visualizations (change masks, overlays)
- Processing statistics

### Comparison Report
- Side-by-side method comparison
- Performance metrics
- Agreement analysis
- Recommendations

### Executive Summary
- High-level findings
- Key statistics
- Critical change areas
- Business implications

## Advanced Usage

### Custom Configuration

```bash
# Create custom configuration
python cli.py --save-config my_config.yaml

# Edit my_config.yaml as needed

# Use custom configuration
python cli.py image1.jpg image2.jpg --config my_config.yaml
```

### Batch Processing

```bash
# Set up for batch processing
python cli.py --save-config batch_config.yaml

# Edit batch_config.yaml:
# - Set logging.level: "WARNING"
# - Set output.save_intermediate: false

# Run batch processing
python cli.py image1.jpg image2.jpg --config batch_config.yaml --quiet
```

### Integration with Other Tools

#### Python Integration
```python
from unified_runner import ChangeDetectionRunner

runner = ChangeDetectionRunner()
results = runner.run_all_implementations("img1.jpg", "img2.jpg")
comparison = runner.compare_results(results)
```

#### Command Line Integration
```bash
# Use in shell scripts
python cli.py "$IMG1" "$IMG2" --save --quiet
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully"
fi
```

## Troubleshooting

### Common Issues

#### Configuration Errors
```bash
# Check configuration
python cli.py --validate-config

# Show current configuration
python cli.py --show-config
```

#### Image Loading Issues
- Ensure images are in supported formats (PNG, JPG, TIFF)
- Check file paths and permissions
- Verify images have the same dimensions

#### Memory Issues
```bash
# Use single implementation for large images
python cli.py image1.jpg image2.jpg -i basic

# Disable progress bars to save memory
python cli.py image1.jpg image2.jpg --no-progress
```

#### Processing Failures
```bash
# Use verbose logging to diagnose
python cli.py image1.jpg image2.jpg --verbose

# Try minimal configuration
python cli.py image1.jpg image2.jpg -i basic --no-reports
```

### Error Messages

#### "Implementation failed"
- Check image compatibility
- Verify sufficient memory
- Try different implementation

#### "No changes detected"
- Adjust minimum area threshold: `--min-area 50`
- Lower confidence threshold: `--confidence-threshold 0.3`
- Check if images are actually different

#### "Report generation failed"
- Check output directory permissions
- Ensure sufficient disk space
- Try without reports: `--no-reports`

## Performance Optimization

### Speed Optimization
1. Use specific implementations instead of all
2. Disable progress bars for batch processing
3. Use WARNING log level for reduced overhead
4. Disable intermediate file saves

### Memory Optimization
1. Process images sequentially, not in parallel
2. Use basic implementation for large images
3. Disable confidence map generation
4. Clear results between batch runs

### Quality Optimization
1. Use appropriate minimum area thresholds
2. Adjust confidence thresholds based on data
3. Use ground truth for validation when available
4. Compare multiple implementations for consensus

## Best Practices

### Image Preparation
- Use images from the same source/sensor
- Ensure proper geometric alignment
- Apply radiometric corrections if needed
- Use consistent image formats

### Parameter Selection
- Start with default parameters
- Adjust based on image characteristics
- Use ground truth for parameter tuning
- Document parameter choices

### Result Validation
- Always review visual outputs
- Compare multiple implementations
- Use accuracy evaluation when possible
- Validate critical findings manually

### Workflow Integration
- Use configuration files for consistency
- Implement proper error handling
- Log all processing steps
- Archive results with metadata

## Example Workflows

### Research Analysis
```bash
# Comprehensive analysis with all features
python cli.py orlando2010.png orlando2023.png \
  --generate-reports \
  --save \
  --verbose \
  --output-dir orlando_analysis \
  --reports-dir orlando_reports
```

### Production Monitoring
```bash
# Automated monitoring with basic implementation
python cli.py current.jpg previous.jpg \
  -i basic \
  --save \
  --quiet \
  --min-area 200 \
  --config production_config.yaml
```

### Quality Assessment
```bash
# Validation with ground truth
python cli.py test_img1.jpg test_img2.jpg \
  -g ground_truth.png \
  --generate-reports \
  --verbose
```

## Support and Maintenance

### System Validation
Run the complete workflow test to validate system functionality:

```bash
python test_complete_workflow.py
```

### Log Analysis
Check system logs for issues:
- Default log location: `results/change_detection.log`
- Use `--verbose` for detailed debugging
- Monitor processing times and memory usage

### Updates and Extensions
The system is designed to be extensible:
- Add new implementations in `implementation_extractors.py`
- Extend report formats in `report_generator.py`
- Add new accuracy metrics in `accuracy_evaluator.py`

## Conclusion

The Change Detection System provides a comprehensive solution for satellite imagery analysis with:
- Multiple detection algorithms
- Standardized reporting
- Accuracy evaluation
- Flexible configuration
- Production-ready CLI

For additional support or feature requests, refer to the system documentation or contact the development team.