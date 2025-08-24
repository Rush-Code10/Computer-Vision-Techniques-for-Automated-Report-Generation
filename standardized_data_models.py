"""
Standardized data structures for change detection results.
This module provides common data models that all implementations will use
to ensure consistent output formats across different approaches.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class ChangeRegion:
    """Represents a single detected change region."""
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area_pixels: int
    area_square_meters: Optional[float] = None
    centroid: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 1.0
    change_type: Optional[str] = None  # 'construction', 'demolition', 'expansion'


@dataclass
class ChangeDetectionResult:
    """Standardized result structure for all change detection implementations."""
    implementation_name: str
    version: str
    timestamp: datetime
    processing_time: float
    
    # Core Results
    change_mask: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    change_regions: List[ChangeRegion] = None
    
    # Metrics
    total_change_area: float = 0.0
    total_change_pixels: int = 0
    num_change_regions: int = 0
    average_confidence: float = 0.0
    
    # Metadata
    parameters: Dict[str, Any] = None
    input_images: Tuple[str, str] = ("", "")
    image_dimensions: Tuple[int, int] = (0, 0)
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.change_regions is None:
            self.change_regions = []
        if self.parameters is None:
            self.parameters = {}
        
        # Calculate metrics if not provided
        if self.total_change_pixels == 0 and self.change_mask is not None:
            self.total_change_pixels = int(np.sum(self.change_mask > 0))
        
        if self.num_change_regions == 0:
            self.num_change_regions = len(self.change_regions)
        
        if self.image_dimensions == (0, 0) and self.change_mask is not None:
            self.image_dimensions = self.change_mask.shape[:2]


@dataclass
class AccuracyMetrics:
    """Accuracy evaluation metrics for change detection results."""
    precision: float
    recall: float
    f1_score: float
    iou: float
    accuracy: float
    specificity: float
    
    # Per-region metrics
    region_precision: List[float] = None
    region_recall: List[float] = None
    
    # Confusion matrix
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.region_precision is None:
            self.region_precision = []
        if self.region_recall is None:
            self.region_recall = []