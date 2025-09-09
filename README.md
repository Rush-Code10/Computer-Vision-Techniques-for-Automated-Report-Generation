# Change Detection System

A comprehensive tool for analyzing infrastructure development and land use changes from satellite imagery using computer vision and machine learning techniques.

## Features

🛰️ **Multi-Algorithm Analysis**: Three different change detection approaches
- Basic Computer Vision (fast, reliable)
- Advanced Computer Vision (sophisticated filtering)  
- Deep Learning Inspired (handles subtle changes)

📊 **Professional Reporting**: Automated PDF report generation with comparative analysis

🎯 **Accuracy Evaluation**: Built-in validation with statistical performance metrics

⚙️ **Production Ready**: CLI interface, configuration management, comprehensive logging

## Quick Start

### Installation
```bash
pip install numpy opencv-python matplotlib scikit-learn reportlab PyYAML
```

### Basic Usage
```bash
# Analyze changes between two satellite images
python cli.py orlando2010.png orlando2023.png

# Generate comprehensive reports
python cli.py orlando2010.png orlando2023.png --generate-reports --save
```

### View Results
- **Results**: JSON files and change masks in `results/` directory
- **Reports**: PDF reports in `reports/` directory
- **Logs**: Processing logs in `results/change_detection.log`

## Example Output

For the Orlando airport analysis (2010 vs 2023):
- **Change Area Detected**: 200K-877K pixels (depending on algorithm)
- **Processing Time**: ~4.4 seconds for complete analysis
- **Key Finding**: Significant airport expansion and development

## Core Components

- `cli.py` - Main command-line interface
- `unified_runner.py` - Core processing engine
- `implementation_extractors.py` - Change detection algorithms
- `accuracy_evaluator.py` - Validation and metrics
- `report_generator.py` - PDF report creation
- `config.yaml` - Configuration settings

## Use Cases

### 🏗️ Infrastructure Monitoring
- Airport expansion tracking
- Urban development analysis
- Construction progress monitoring

### 🌍 Environmental Analysis
- Deforestation detection
- Urban sprawl measurement
- Land use change assessment

### 📈 Business Intelligence
- Market expansion analysis
- Competitor facility monitoring
- Investment opportunity identification

## System Requirements

- **Python**: 3.7 or higher
- **Memory**: 512MB+ (depends on image size)
- **Storage**: 100MB+ for results and reports
- **OS**: Windows, macOS, or Linux

## Performance

- **Speed**: 0.06-0.49 seconds per algorithm
- **Memory**: ~336MB peak usage for complete workflow
- **Reliability**: 100% test success rate

## Getting Help

```bash
# Show all available options
python cli.py --help

# Validate your setup
python cli.py --validate-config
```

---

**Status**: ✅ Production Ready | **Version**: 1.0