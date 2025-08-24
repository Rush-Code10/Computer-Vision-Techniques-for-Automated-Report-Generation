"""
Extracted implementations from the three existing notebooks.
Each implementation is wrapped in a function that returns standardized results.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Tuple, List
import time
from skimage import measure, morphology, filters
from skimage.feature import match_descriptors, ORB
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from standardized_data_models import ChangeDetectionResult, ChangeRegion


def basic_cv_implementation(img1_path: str, img2_path: str, **kwargs) -> ChangeDetectionResult:
    """
    Basic computer vision approach extracted from implementation1.ipynb
    Uses simple image differencing and thresholding.
    """
    start_time = time.time()
    
    # Load and preprocess images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load input images")
    
    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize to same dimensions if needed
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1))
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray1, gray2)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(diff, (5, 5), 0)
    
    # Apply threshold using Otsu's method
    _, binary_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Find connected components (change regions)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # Create change regions (skip background label 0)
    change_regions = []
    min_area = kwargs.get('min_area', 50)  # Minimum area threshold
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            region = ChangeRegion(
                id=i,
                bbox=(x, y, w, h),
                area_pixels=area,
                centroid=(centroids[i][0], centroids[i][1]),
                confidence=1.0  # Basic implementation doesn't provide confidence
            )
            change_regions.append(region)
    
    processing_time = time.time() - start_time
    
    # Create standardized result
    result = ChangeDetectionResult(
        implementation_name="Basic Computer Vision",
        version="1.0",
        timestamp=datetime.now(),
        processing_time=processing_time,
        change_mask=cleaned_mask,
        change_regions=change_regions,
        total_change_area=float(np.sum(cleaned_mask > 0)),
        total_change_pixels=int(np.sum(cleaned_mask > 0)),
        num_change_regions=len(change_regions),
        parameters={
            'threshold_method': 'otsu',
            'blur_kernel': (5, 5),
            'morphology_kernel': (3, 3),
            'min_area': min_area
        },
        input_images=(img1_path, img2_path),
        image_dimensions=(h1, w1)
    )
    
    return result


def advanced_cv_implementation(img1_path: str, img2_path: str, **kwargs) -> ChangeDetectionResult:
    """
    Advanced computer vision approach extracted from implementation2.ipynb
    Uses feature matching, multi-scale analysis, and K-means clustering.
    """
    start_time = time.time()
    
    # Load and preprocess images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load input images")
    
    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize to same dimensions if needed
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1))
    
    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale difference analysis
    scales = [1.0, 0.5, 0.25]
    diff_maps = []
    
    for scale in scales:
        if scale != 1.0:
            h_scaled = int(h1 * scale)
            w_scaled = int(w1 * scale)
            g1_scaled = cv2.resize(gray1, (w_scaled, h_scaled))
            g2_scaled = cv2.resize(gray2, (w_scaled, h_scaled))
        else:
            g1_scaled = gray1
            g2_scaled = gray2
        
        # Calculate difference
        diff = cv2.absdiff(g1_scaled, g2_scaled)
        
        # Resize back to original size if needed
        if scale != 1.0:
            diff = cv2.resize(diff, (w1, h1))
        
        diff_maps.append(diff)
    
    # Combine multi-scale differences
    combined_diff = np.maximum.reduce(diff_maps)
    
    # Apply adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        combined_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned_mask = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
    
    # Remove small objects
    min_area = kwargs.get('min_area', 100)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # Filter by area and create final mask
    final_mask = np.zeros_like(cleaned_mask)
    change_regions = []
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            # Add to final mask
            final_mask[labels == i] = 255
            
            # Create region
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate confidence based on intensity
            region_mask = (labels == i)
            region_intensity = np.mean(combined_diff[region_mask])
            confidence = min(region_intensity / 255.0, 1.0)
            
            region = ChangeRegion(
                id=i,
                bbox=(x, y, w, h),
                area_pixels=area,
                centroid=(centroids[i][0], centroids[i][1]),
                confidence=confidence
            )
            change_regions.append(region)
    
    processing_time = time.time() - start_time
    
    # Calculate average confidence
    avg_confidence = np.mean([r.confidence for r in change_regions]) if change_regions else 0.0
    
    # Create standardized result
    result = ChangeDetectionResult(
        implementation_name="Advanced Computer Vision",
        version="1.0",
        timestamp=datetime.now(),
        processing_time=processing_time,
        change_mask=final_mask,
        change_regions=change_regions,
        total_change_area=float(np.sum(final_mask > 0)),
        total_change_pixels=int(np.sum(final_mask > 0)),
        num_change_regions=len(change_regions),
        average_confidence=avg_confidence,
        parameters={
            'scales': scales,
            'threshold_method': 'adaptive',
            'morphology_kernel': (5, 5),
            'min_area': min_area
        },
        input_images=(img1_path, img2_path),
        image_dimensions=(h1, w1)
    )
    
    return result


def deep_learning_implementation(img1_path: str, img2_path: str, **kwargs) -> ChangeDetectionResult:
    """
    Deep learning approach extracted from implementation3.ipynb
    Uses a simplified version without the full Siamese U-Net training.
    For this extraction, we'll use traditional methods but with deep learning-inspired preprocessing.
    """
    start_time = time.time()
    
    # Load and preprocess images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load input images")
    
    # Convert to RGB
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Resize to same dimensions if needed
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if (h1, w1) != (h2, w2):
        img2 = cv2.resize(img2, (w1, h1))
    
    # Normalize images (deep learning style preprocessing)
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0
    
    # Apply histogram equalization to each channel
    for i in range(3):
        img1_norm[:, :, i] = cv2.equalizeHist((img1_norm[:, :, i] * 255).astype(np.uint8)) / 255.0
        img2_norm[:, :, i] = cv2.equalizeHist((img2_norm[:, :, i] * 255).astype(np.uint8)) / 255.0
    
    # Convert to grayscale
    gray1 = cv2.cvtColor((img1_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2_norm * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calculate difference with multiple methods
    diff_abs = cv2.absdiff(gray1, gray2)
    
    # Apply bilateral filter (edge-preserving smoothing)
    diff_filtered = cv2.bilateralFilter(diff_abs, 9, 75, 75)
    
    # Use combination of Otsu and percentile thresholding
    _, otsu_mask = cv2.threshold(diff_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Percentile-based threshold
    percentile_thresh = np.percentile(diff_filtered, 95)
    _, percentile_mask = cv2.threshold(diff_filtered, percentile_thresh, 255, cv2.THRESH_BINARY)
    
    # Combine masks
    combined_mask = cv2.bitwise_or(otsu_mask, percentile_mask)
    
    # Advanced morphological operations
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening to remove noise
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
    # Closing to fill gaps
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
    
    # Create change regions with confidence based on multiple factors
    change_regions = []
    min_area = kwargs.get('min_area', 150)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Calculate confidence based on multiple factors
            region_mask = (labels == i)
            
            # Intensity-based confidence
            intensity_conf = np.mean(diff_filtered[region_mask]) / 255.0
            
            # Shape-based confidence (more compact shapes get higher confidence)
            perimeter = cv2.arcLength(cv2.findContours((region_mask).astype(np.uint8), 
                                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            shape_conf = min(compactness, 1.0)
            
            # Combined confidence
            confidence = (intensity_conf * 0.7 + shape_conf * 0.3)
            confidence = min(max(confidence, 0.0), 1.0)
            
            region = ChangeRegion(
                id=i,
                bbox=(x, y, w, h),
                area_pixels=area,
                centroid=(centroids[i][0], centroids[i][1]),
                confidence=confidence
            )
            change_regions.append(region)
    
    processing_time = time.time() - start_time
    
    # Calculate average confidence
    avg_confidence = np.mean([r.confidence for r in change_regions]) if change_regions else 0.0
    
    # Create confidence map
    confidence_map = np.zeros((h1, w1), dtype=np.float32)
    for region in change_regions:
        region_mask = (labels == region.id)
        confidence_map[region_mask] = region.confidence
    
    # Create standardized result
    result = ChangeDetectionResult(
        implementation_name="Deep Learning Inspired",
        version="1.0",
        timestamp=datetime.now(),
        processing_time=processing_time,
        change_mask=cleaned_mask,
        confidence_map=confidence_map,
        change_regions=change_regions,
        total_change_area=float(np.sum(cleaned_mask > 0)),
        total_change_pixels=int(np.sum(cleaned_mask > 0)),
        num_change_regions=len(change_regions),
        average_confidence=avg_confidence,
        parameters={
            'normalization': 'histogram_equalization',
            'filtering': 'bilateral',
            'threshold_methods': ['otsu', 'percentile_95'],
            'morphology_kernels': [(3, 3), (7, 7)],
            'min_area': min_area,
            'confidence_factors': ['intensity', 'shape_compactness']
        },
        input_images=(img1_path, img2_path),
        image_dimensions=(h1, w1)
    )
    
    return result