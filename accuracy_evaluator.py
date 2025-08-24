"""
Accuracy evaluation module for change detection implementations.
Provides comprehensive accuracy assessment capabilities including ground truth evaluation
and inter-method agreement analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from scipy.spatial.distance import jaccard
from standardized_data_models import ChangeDetectionResult, AccuracyMetrics


class AccuracyEvaluator:
    """Provides accuracy evaluation capabilities for change detection results."""
    
    def __init__(self):
        pass
    
    def evaluate_with_ground_truth(self, result: ChangeDetectionResult, 
                                 ground_truth: np.ndarray) -> AccuracyMetrics:
        """
        Evaluate accuracy metrics when ground truth is available.
        
        Args:
            result: Change detection result to evaluate
            ground_truth: Binary ground truth mask (0=no change, 255=change)
            
        Returns:
            AccuracyMetrics object with calculated metrics
        """
        # Ensure both masks are binary
        pred_mask = (result.change_mask > 0).astype(np.uint8)
        gt_mask = (ground_truth > 0).astype(np.uint8)
        
        # Ensure same dimensions
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(f"Prediction mask shape {pred_mask.shape} doesn't match ground truth shape {gt_mask.shape}")
        
        # Calculate confusion matrix components
        tp = np.sum((pred_mask == 1) & (gt_mask == 1))  # True positives
        fp = np.sum((pred_mask == 1) & (gt_mask == 0))  # False positives
        tn = np.sum((pred_mask == 0) & (gt_mask == 0))  # True negatives
        fn = np.sum((pred_mask == 0) & (gt_mask == 1))  # False negatives
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # Calculate IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        
        return AccuracyMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            iou=iou,
            accuracy=accuracy,
            specificity=specificity,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn)
        )
    
    def evaluate_inter_method_agreement(self, results: List[ChangeDetectionResult]) -> Dict[str, Any]:
        """
        Evaluate agreement between multiple change detection methods.
        
        Args:
            results: List of change detection results to compare
            
        Returns:
            Dictionary containing agreement metrics and analysis
        """
        if len(results) < 2:
            return {"error": "Need at least 2 results for agreement analysis"}
        
        # Extract binary masks
        masks = []
        names = []
        for result in results:
            binary_mask = (result.change_mask > 0).astype(np.uint8)
            masks.append(binary_mask)
            names.append(result.implementation_name)
        
        # Ensure all masks have the same shape
        base_shape = masks[0].shape
        for i, mask in enumerate(masks):
            if mask.shape != base_shape:
                raise ValueError(f"Mask {i} shape {mask.shape} doesn't match base shape {base_shape}")
        
        agreement_analysis = {
            "num_methods": len(results),
            "method_names": names,
            "pairwise_agreements": {},
            "overall_agreement": {},
            "consensus_analysis": {}
        }
        
        # Calculate pairwise agreements
        pairwise_ious = []
        pairwise_jaccards = []
        
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                name_i = names[i]
                name_j = names[j]
                mask_i = masks[i]
                mask_j = masks[j]
                
                # Calculate IoU
                intersection = np.sum((mask_i == 1) & (mask_j == 1))
                union = np.sum((mask_i == 1) | (mask_j == 1))
                iou = intersection / union if union > 0 else 0.0
                
                # Calculate Jaccard similarity
                jaccard_sim = jaccard_score(mask_i.flatten(), mask_j.flatten(), average='binary')
                
                # Calculate pixel agreement
                pixel_agreement = np.sum(mask_i == mask_j) / mask_i.size
                
                pair_key = f"{name_i}_vs_{name_j}"
                agreement_analysis["pairwise_agreements"][pair_key] = {
                    "iou": iou,
                    "jaccard_similarity": jaccard_sim,
                    "pixel_agreement": pixel_agreement
                }
                
                pairwise_ious.append(iou)
                pairwise_jaccards.append(jaccard_sim)
        
        # Overall agreement statistics
        agreement_analysis["overall_agreement"] = {
            "mean_iou": np.mean(pairwise_ious),
            "std_iou": np.std(pairwise_ious),
            "min_iou": np.min(pairwise_ious),
            "max_iou": np.max(pairwise_ious),
            "mean_jaccard": np.mean(pairwise_jaccards),
            "std_jaccard": np.std(pairwise_jaccards)
        }
        
        # Consensus analysis
        consensus_mask = np.zeros_like(masks[0], dtype=np.float32)
        for mask in masks:
            consensus_mask += mask.astype(np.float32)
        
        # Normalize to get agreement level per pixel
        consensus_mask /= len(masks)
        
        # Calculate consensus statistics
        full_agreement_pixels = np.sum(consensus_mask == 1.0)  # All methods agree on change
        no_agreement_pixels = np.sum(consensus_mask == 0.0)    # All methods agree on no change
        partial_agreement_pixels = np.sum((consensus_mask > 0) & (consensus_mask < 1.0))
        
        total_pixels = consensus_mask.size
        
        agreement_analysis["consensus_analysis"] = {
            "full_agreement_change_pixels": int(full_agreement_pixels),
            "full_agreement_no_change_pixels": int(no_agreement_pixels),
            "partial_agreement_pixels": int(partial_agreement_pixels),
            "full_agreement_percentage": (full_agreement_pixels + no_agreement_pixels) / total_pixels * 100,
            "consensus_mask_mean": float(np.mean(consensus_mask)),
            "consensus_mask_std": float(np.std(consensus_mask))
        }
        
        # Agreement level classification
        mean_iou = agreement_analysis["overall_agreement"]["mean_iou"]
        if mean_iou >= 0.7:
            agreement_level = "High"
        elif mean_iou >= 0.4:
            agreement_level = "Medium"
        else:
            agreement_level = "Low"
        
        agreement_analysis["agreement_level"] = agreement_level
        
        return agreement_analysis
    
    def calculate_confidence_intervals(self, results: List[ChangeDetectionResult], 
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calculate confidence intervals for key metrics across multiple results.
        
        Args:
            results: List of change detection results
            confidence_level: Confidence level for intervals (default: 0.95)
            
        Returns:
            Dictionary containing confidence intervals for various metrics
        """
        if len(results) < 2:
            return {"error": "Need at least 2 results for confidence intervals"}
        
        # Extract metrics
        change_pixels = [r.total_change_pixels for r in results]
        num_regions = [r.num_change_regions for r in results]
        processing_times = [r.processing_time for r in results]
        confidences = [r.average_confidence for r in results if r.average_confidence > 0]
        
        def calculate_ci(data, confidence_level):
            """Calculate confidence interval for a dataset."""
            if len(data) < 2:
                return {"mean": 0, "ci_lower": 0, "ci_upper": 0, "std": 0}
            
            mean = np.mean(data)
            std = np.std(data, ddof=1)  # Sample standard deviation
            n = len(data)
            
            # Use t-distribution for small samples
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1)
            margin_error = t_value * (std / np.sqrt(n))
            
            return {
                "mean": float(mean),
                "std": float(std),
                "ci_lower": float(mean - margin_error),
                "ci_upper": float(mean + margin_error),
                "margin_error": float(margin_error)
            }
        
        confidence_intervals = {
            "confidence_level": confidence_level,
            "sample_size": len(results),
            "change_pixels": calculate_ci(change_pixels, confidence_level),
            "num_regions": calculate_ci(num_regions, confidence_level),
            "processing_time": calculate_ci(processing_times, confidence_level)
        }
        
        if confidences:
            confidence_intervals["average_confidence"] = calculate_ci(confidences, confidence_level)
        
        return confidence_intervals
    
    def generate_accuracy_report(self, results: List[ChangeDetectionResult], 
                               ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive accuracy report.
        
        Args:
            results: List of change detection results
            ground_truth: Optional ground truth mask for accuracy evaluation
            
        Returns:
            Dictionary containing comprehensive accuracy analysis
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_methods": len(results),
            "method_names": [r.implementation_name for r in results]
        }
        
        # Ground truth evaluation if available
        if ground_truth is not None:
            report["ground_truth_evaluation"] = {}
            for result in results:
                try:
                    metrics = self.evaluate_with_ground_truth(result, ground_truth)
                    report["ground_truth_evaluation"][result.implementation_name] = {
                        "precision": metrics.precision,
                        "recall": metrics.recall,
                        "f1_score": metrics.f1_score,
                        "iou": metrics.iou,
                        "accuracy": metrics.accuracy,
                        "specificity": metrics.specificity,
                        "confusion_matrix": {
                            "true_positives": metrics.true_positives,
                            "false_positives": metrics.false_positives,
                            "true_negatives": metrics.true_negatives,
                            "false_negatives": metrics.false_negatives
                        }
                    }
                except Exception as e:
                    report["ground_truth_evaluation"][result.implementation_name] = {
                        "error": str(e)
                    }
        
        # Inter-method agreement analysis
        if len(results) >= 2:
            try:
                agreement_analysis = self.evaluate_inter_method_agreement(results)
                report["inter_method_agreement"] = agreement_analysis
            except Exception as e:
                report["inter_method_agreement"] = {"error": str(e)}
        
        # Confidence intervals
        if len(results) >= 2:
            try:
                confidence_intervals = self.calculate_confidence_intervals(results)
                report["confidence_intervals"] = confidence_intervals
            except Exception as e:
                report["confidence_intervals"] = {"error": str(e)}
        
        return report
    
    def print_accuracy_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of the accuracy report."""
        print("\n" + "=" * 60)
        print("ACCURACY EVALUATION SUMMARY")
        print("=" * 60)
        
        print(f"Number of methods evaluated: {report['num_methods']}")
        print(f"Methods: {', '.join(report['method_names'])}")
        
        # Ground truth evaluation
        if "ground_truth_evaluation" in report:
            print(f"\n📊 GROUND TRUTH EVALUATION:")
            print("-" * 40)
            for method, metrics in report["ground_truth_evaluation"].items():
                if "error" in metrics:
                    print(f"{method}: ❌ {metrics['error']}")
                else:
                    print(f"{method}:")
                    print(f"  Precision: {metrics['precision']:.3f}")
                    print(f"  Recall:    {metrics['recall']:.3f}")
                    print(f"  F1-Score:  {metrics['f1_score']:.3f}")
                    print(f"  IoU:       {metrics['iou']:.3f}")
                    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        
        # Inter-method agreement
        if "inter_method_agreement" in report and "error" not in report["inter_method_agreement"]:
            agreement = report["inter_method_agreement"]
            print(f"\n🤝 INTER-METHOD AGREEMENT:")
            print("-" * 40)
            print(f"Agreement Level: {agreement['agreement_level']}")
            print(f"Mean IoU: {agreement['overall_agreement']['mean_iou']:.3f}")
            print(f"Mean Jaccard: {agreement['overall_agreement']['mean_jaccard']:.3f}")
            
            consensus = agreement["consensus_analysis"]
            print(f"Full Agreement: {consensus['full_agreement_percentage']:.1f}% of pixels")
        
        # Confidence intervals
        if "confidence_intervals" in report and "error" not in report["confidence_intervals"]:
            ci = report["confidence_intervals"]
            print(f"\n📈 CONFIDENCE INTERVALS ({ci['confidence_level']*100:.0f}%):")
            print("-" * 40)
            
            if "change_pixels" in ci:
                cp = ci["change_pixels"]
                print(f"Change Pixels: {cp['mean']:.0f} ± {cp['margin_error']:.0f}")
            
            if "num_regions" in ci:
                nr = ci["num_regions"]
                print(f"Num Regions: {nr['mean']:.1f} ± {nr['margin_error']:.1f}")
            
            if "processing_time" in ci:
                pt = ci["processing_time"]
                print(f"Processing Time: {pt['mean']:.3f} ± {pt['margin_error']:.3f} seconds")