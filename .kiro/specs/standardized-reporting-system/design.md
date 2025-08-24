# Design Document

## Overview

The standardized reporting system will create a unified framework for processing satellite imagery change detection results from multiple implementations. The system will provide consistent output formats, accuracy evaluation metrics, and comprehensive reporting capabilities while maintaining the flexibility to work with different algorithmic approaches.

## Architecture

The system follows a modular architecture with the following key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Unified Interface                        │
├─────────────────────────────────────────────────────────────┤
│  Implementation Manager  │  Report Generator  │  Evaluator  │
├─────────────────────────────────────────────────────────────┤
│           Standardized Data Models & Schemas               │
├─────────────────────────────────────────────────────────────┤
│  CV Basic  │  CV Advanced  │  Deep Learning  │  Future...   │
└─────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Standardized Data Models

#### ChangeDetectionResult
```python
@dataclass
class ChangeDetectionResult:
    implementation_name: str
    version: str
    timestamp: datetime
    processing_time: float
    
    # Core Results
    change_mask: np.ndarray
    confidence_map: Optional[np.ndarray]
    change_regions: List[ChangeRegion]
    
    # Metrics
    total_change_area: float
    total_change_pixels: int
    num_change_regions: int
    average_confidence: float
    
    # Metadata
    parameters: Dict[str, Any]
    input_images: Tuple[str, str]
    image_dimensions: Tuple[int, int]
```

#### ChangeRegion
```python
@dataclass
class ChangeRegion:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area_pixels: int
    area_square_meters: Optional[float]
    centroid: Tuple[float, float]
    confidence: float
    change_type: Optional[str]  # 'construction', 'demolition', 'expansion'
```

#### AccuracyMetrics
```python
@dataclass
class AccuracyMetrics:
    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    specificity: float
    
    # Per-region metrics
    region_precision: List[float]
    region_recall: List[float]
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
```

### 2. Implementation Manager

The Implementation Manager handles the execution and standardization of different change detection approaches:

```python
class ImplementationManager:
    def __init__(self):
        self.implementations = {
            'cv_basic': BasicCVImplementation(),
            'cv_advanced': AdvancedCVImplementation(), 
            'deep_learning': DeepLearningImplementation()
        }
    
    def run_implementation(self, name: str, config: Dict) -> ChangeDetectionResult
    def run_all_implementations(self, config: Dict) -> List[ChangeDetectionResult]
    def standardize_output(self, raw_result: Any, impl_name: str) -> ChangeDetectionResult
```

### 3. Report Generator

Generates standardized reports in multiple formats:

```python
class ReportGenerator:
    def generate_individual_report(self, result: ChangeDetectionResult) -> Report
    def generate_comparison_report(self, results: List[ChangeDetectionResult]) -> Report
    def generate_executive_summary(self, results: List[ChangeDetectionResult]) -> Report
    def export_report(self, report: Report, format: str, output_path: str)
```

### 4. Accuracy Evaluator

Provides comprehensive accuracy assessment capabilities:

```python
class AccuracyEvaluator:
    def evaluate_with_ground_truth(self, result: ChangeDetectionResult, 
                                 ground_truth: np.ndarray) -> AccuracyMetrics
    def evaluate_inter_method_agreement(self, results: List[ChangeDetectionResult]) -> Dict
    def calculate_confidence_intervals(self, results: List[ChangeDetectionResult]) -> Dict
    def generate_accuracy_visualizations(self, metrics: AccuracyMetrics) -> List[Figure]
```

## Data Models

### Configuration Schema
```yaml
# config.yaml
input:
  image1_path: str
  image2_path: str
  ground_truth_path: Optional[str]
  
processing:
  implementations: List[str]  # ['cv_basic', 'cv_advanced', 'deep_learning']
  parallel_execution: bool
  
output:
  base_directory: str
  formats: List[str]  # ['pdf', 'html', 'json']
  include_visualizations: bool
  
evaluation:
  calculate_accuracy: bool
  confidence_threshold: float
  minimum_region_size: int
```

### Report Structure
```
reports/
├── timestamp_YYYYMMDD_HHMMSS/
│   ├── individual/
│   │   ├── cv_basic_report.pdf
│   │   ├── cv_advanced_report.pdf
│   │   └── deep_learning_report.pdf
│   ├── comparison/
│   │   ├── comparison_report.pdf
│   │   ├── accuracy_analysis.pdf
│   │   └── executive_summary.pdf
│   ├── data/
│   │   ├── results.json
│   │   ├── accuracy_metrics.json
│   │   └── configuration.yaml
│   └── visualizations/
│       ├── change_masks/
│       ├── confidence_maps/
│       └── accuracy_plots/
```

## Error Handling

The system implements comprehensive error handling:

1. **Input Validation**: Verify image formats, dimensions, and accessibility
2. **Implementation Failures**: Graceful handling when individual implementations fail
3. **Resource Management**: Memory and processing time limits
4. **Output Validation**: Ensure generated reports meet quality standards

## Testing Strategy

### Unit Testing
- Test each component independently
- Mock external dependencies
- Validate data model serialization/deserialization

### Integration Testing  
- Test end-to-end workflows
- Validate report generation with real data
- Test accuracy evaluation with known ground truth

### Performance Testing
- Benchmark processing times for different image sizes
- Memory usage profiling
- Parallel execution efficiency

### Validation Testing
- Compare outputs with manual analysis
- Cross-validate accuracy metrics
- Test with diverse satellite imagery datasets