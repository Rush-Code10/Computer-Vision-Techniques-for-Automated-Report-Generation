"""
Report generation module for change detection implementations.
Creates standardized PDF reports for individual methods and comparison reports.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import cv2

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.platypus.flowables import KeepTogether
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

from standardized_data_models import ChangeDetectionResult, AccuracyMetrics
from visualization_components import VisualizationComponents


class ReportGenerator:
    """Generates standardized reports for change detection results."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.viz_components = VisualizationComponents()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for reports."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=HexColor('#2E86AB'),
            alignment=1  # Center alignment
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=HexColor('#A23B72'),
            borderWidth=1,
            borderColor=HexColor('#A23B72'),
            borderPadding=5
        ))
        
        # Subsection header style
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=8,
            spaceBefore=12,
            textColor=HexColor('#F18F01')
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            bulletIndent=10
        ))
    
    def generate_individual_report(self, result: ChangeDetectionResult, 
                                 accuracy_metrics: Optional[AccuracyMetrics] = None,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate a standardized PDF report for a single implementation.
        
        Args:
            result: Change detection result to report on
            accuracy_metrics: Optional accuracy metrics if ground truth available
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.implementation_name.lower().replace(' ', '_')}_report_{timestamp}.pdf"
            output_path = os.path.join("reports", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        title = f"{result.implementation_name} Change Detection Report"
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        summary_text = self._generate_executive_summary(result, accuracy_metrics)
        story.append(Paragraph(summary_text, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Method Description
        story.append(Paragraph("Method Description", self.styles['SectionHeader']))
        method_desc = self._get_method_description(result.implementation_name)
        story.append(Paragraph(method_desc, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Results Summary
        story.append(Paragraph("Results Summary", self.styles['SectionHeader']))
        results_table = self._create_results_table(result)
        story.append(results_table)
        story.append(Spacer(1, 15))
        
        # Change Statistics
        story.append(Paragraph("Change Statistics", self.styles['SectionHeader']))
        stats_content = self._generate_change_statistics(result)
        for item in stats_content:
            story.append(item)
        story.append(Spacer(1, 15))
        
        # Accuracy Metrics (if available)
        if accuracy_metrics:
            story.append(Paragraph("Accuracy Evaluation", self.styles['SectionHeader']))
            accuracy_table = self._create_accuracy_table(accuracy_metrics)
            story.append(accuracy_table)
            story.append(Spacer(1, 15))
        
        # Technical Details
        story.append(Paragraph("Technical Details", self.styles['SectionHeader']))
        tech_table = self._create_technical_details_table(result)
        story.append(tech_table)
        story.append(Spacer(1, 15))
        
        # Visualizations section
        story.append(Paragraph("Visualizations", self.styles['SectionHeader']))
        viz_path = self._generate_visualizations(result, accuracy_metrics)
        if viz_path and os.path.exists(viz_path):
            story.append(Image(viz_path, width=6*inch, height=4*inch))
        story.append(Spacer(1, 15))
        
        # Footer
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"Individual report generated: {output_path}")
        return output_path
    
    def generate_comparison_report(self, results: List[ChangeDetectionResult],
                                 accuracy_report: Optional[Dict[str, Any]] = None,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate a comparison report showing all methods side-by-side.
        
        Args:
            results: List of change detection results to compare
            accuracy_report: Optional accuracy evaluation report
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{timestamp}.pdf"
            output_path = os.path.join("reports", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Change Detection Methods Comparison Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        comparison_summary = self._generate_comparison_summary(results, accuracy_report)
        story.append(Paragraph(comparison_summary, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Methods Overview
        story.append(Paragraph("Methods Overview", self.styles['SectionHeader']))
        methods_table = self._create_methods_comparison_table(results)
        story.append(methods_table)
        story.append(Spacer(1, 15))
        
        # Performance Comparison
        story.append(Paragraph("Performance Comparison", self.styles['SectionHeader']))
        performance_table = self._create_performance_comparison_table(results)
        story.append(performance_table)
        story.append(Spacer(1, 15))
        
        # Accuracy Comparison (if available)
        if accuracy_report and "ground_truth_evaluation" in accuracy_report:
            story.append(Paragraph("Accuracy Comparison", self.styles['SectionHeader']))
            accuracy_table = self._create_accuracy_comparison_table(accuracy_report["ground_truth_evaluation"])
            story.append(accuracy_table)
            story.append(Spacer(1, 15))
        
        # Agreement Analysis (if available)
        if accuracy_report and "inter_method_agreement" in accuracy_report:
            story.append(Paragraph("Inter-Method Agreement", self.styles['SectionHeader']))
            agreement_content = self._generate_agreement_analysis(accuracy_report["inter_method_agreement"])
            for item in agreement_content:
                story.append(item)
            story.append(Spacer(1, 15))
        
        # Recommendations
        story.append(Paragraph("Recommendations", self.styles['SectionHeader']))
        recommendations = self._generate_recommendations(results, accuracy_report)
        story.append(Paragraph(recommendations, self.styles['Normal']))
        story.append(Spacer(1, 15))
        
        # Comparison Visualizations
        story.append(PageBreak())
        story.append(Paragraph("Comparison Visualizations", self.styles['SectionHeader']))
        viz_path = self._generate_comparison_visualizations(results, accuracy_report)
        if viz_path and os.path.exists(viz_path):
            story.append(Image(viz_path, width=7*inch, height=5*inch))
        
        # Footer
        footer_text = f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Spacer(1, 20))
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"Comparison report generated: {output_path}")
        return output_path
    
    def _generate_executive_summary(self, result: ChangeDetectionResult, 
                                   accuracy_metrics: Optional[AccuracyMetrics] = None) -> str:
        """Generate executive summary text for individual report."""
        change_area_pct = (result.total_change_pixels / (result.image_dimensions[0] * result.image_dimensions[1])) * 100
        
        summary = f"""
        The {result.implementation_name} method processed satellite imagery and detected {result.total_change_pixels:,} 
        pixels of change across {result.num_change_regions} distinct regions, representing {change_area_pct:.2f}% 
        of the total image area. Processing completed in {result.processing_time:.3f} seconds.
        """
        
        if accuracy_metrics:
            summary += f"""
            
            Accuracy evaluation shows a precision of {accuracy_metrics.precision:.3f}, recall of {accuracy_metrics.recall:.3f}, 
            and F1-score of {accuracy_metrics.f1_score:.3f}. The method achieved an IoU (Intersection over Union) 
            of {accuracy_metrics.iou:.3f} when compared to ground truth data.
            """
        
        if result.average_confidence > 0:
            summary += f" The average confidence score for detected changes was {result.average_confidence:.3f}."
        
        return summary.strip()
    
    def _get_method_description(self, implementation_name: str) -> str:
        """Get description for each implementation method."""
        descriptions = {
            "Basic Computer Vision": """
            This method uses traditional computer vision techniques including image differencing, 
            Gaussian blur filtering, and morphological operations to detect changes between two 
            satellite images. It applies threshold-based segmentation and contour detection to 
            identify change regions. This approach is computationally efficient and provides 
            reliable results for clear, high-contrast changes.
            """,
            "Advanced Computer Vision": """
            This method employs sophisticated computer vision algorithms including adaptive 
            thresholding, multi-scale analysis, and advanced morphological operations. It uses 
            edge detection, feature matching, and statistical analysis to improve change detection 
            accuracy. The method includes noise reduction techniques and region growing algorithms 
            to better handle complex change patterns and varying lighting conditions.
            """,
            "Deep Learning Inspired": """
            This method incorporates deep learning-inspired techniques including feature extraction 
            using convolutional operations, multi-level analysis, and confidence scoring. It applies 
            learned patterns and statistical modeling to identify changes while providing confidence 
            estimates for each detection. The approach balances computational efficiency with the 
            sophisticated pattern recognition capabilities inspired by neural network architectures.
            """
        }
        
        return descriptions.get(implementation_name, "Method description not available.").strip()
    
    def _create_results_table(self, result: ChangeDetectionResult) -> Table:
        """Create a table with key results metrics."""
        data = [
            ['Metric', 'Value'],
            ['Total Change Pixels', f"{result.total_change_pixels:,}"],
            ['Number of Change Regions', str(result.num_change_regions)],
            ['Total Change Area', f"{result.total_change_area:.2f} sq units"],
            ['Processing Time', f"{result.processing_time:.3f} seconds"],
            ['Image Dimensions', f"{result.image_dimensions[0]} x {result.image_dimensions[1]}"],
            ['Average Confidence', f"{result.average_confidence:.3f}" if result.average_confidence > 0 else "N/A"]
        ]
        
        table = Table(data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_change_statistics(self, result: ChangeDetectionResult) -> List:
        """Generate change statistics content."""
        content = []
        
        if result.change_regions:
            # Region size statistics
            region_sizes = [r.area_pixels for r in result.change_regions]
            content.append(Paragraph("Region Size Analysis:", self.styles['SubsectionHeader']))
            
            stats_text = f"""
            • Largest region: {max(region_sizes):,} pixels<br/>
            • Smallest region: {min(region_sizes):,} pixels<br/>
            • Average region size: {np.mean(region_sizes):,.0f} pixels<br/>
            • Median region size: {np.median(region_sizes):,.0f} pixels
            """
            content.append(Paragraph(stats_text, self.styles['MetricStyle']))
            
            # Top 5 largest regions
            if len(result.change_regions) > 1:
                sorted_regions = sorted(result.change_regions, key=lambda r: r.area_pixels, reverse=True)
                content.append(Paragraph("Top 5 Largest Change Regions:", self.styles['SubsectionHeader']))
                
                region_data = [['Region ID', 'Area (pixels)', 'Confidence', 'Center (x, y)']]
                for i, region in enumerate(sorted_regions[:5]):
                    region_data.append([
                        str(region.id),
                        f"{region.area_pixels:,}",
                        f"{region.confidence:.3f}" if region.confidence > 0 else "N/A",
                        f"({region.centroid[0]:.0f}, {region.centroid[1]:.0f})"
                    ])
                
                region_table = Table(region_data, colWidths=[1*inch, 1.5*inch, 1*inch, 1.5*inch])
                region_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                content.append(region_table)
        else:
            content.append(Paragraph("No individual change regions were identified.", self.styles['Normal']))
        
        return content
    
    def _create_accuracy_table(self, accuracy_metrics: AccuracyMetrics) -> Table:
        """Create accuracy metrics table."""
        data = [
            ['Metric', 'Value'],
            ['Precision', f"{accuracy_metrics.precision:.4f}"],
            ['Recall', f"{accuracy_metrics.recall:.4f}"],
            ['F1-Score', f"{accuracy_metrics.f1_score:.4f}"],
            ['IoU', f"{accuracy_metrics.iou:.4f}"],
            ['Accuracy', f"{accuracy_metrics.accuracy:.4f}"],
            ['Specificity', f"{accuracy_metrics.specificity:.4f}"]
        ]
        
        table = Table(data, colWidths=[2*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_technical_details_table(self, result: ChangeDetectionResult) -> Table:
        """Create technical details table."""
        data = [
            ['Parameter', 'Value'],
            ['Implementation', result.implementation_name],
            ['Version', result.version],
            ['Timestamp', result.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Input Image 1', os.path.basename(result.input_images[0]) if result.input_images[0] else "N/A"],
            ['Input Image 2', os.path.basename(result.input_images[1]) if result.input_images[1] else "N/A"]
        ]
        
        # Add parameters if available
        if result.parameters:
            for key, value in result.parameters.items():
                data.append([str(key), str(value)])
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_comparison_summary(self, results: List[ChangeDetectionResult], 
                                   accuracy_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate executive summary for comparison report."""
        if not results:
            return "No results available for comparison."
        
        method_names = [r.implementation_name for r in results]
        processing_times = [r.processing_time for r in results]
        change_pixels = [r.total_change_pixels for r in results]
        
        fastest_idx = processing_times.index(min(processing_times))
        most_changes_idx = change_pixels.index(max(change_pixels))
        
        summary = f"""
        This report compares {len(results)} change detection methods: {', '.join(method_names)}. 
        The fastest method was {results[fastest_idx].implementation_name} ({processing_times[fastest_idx]:.3f}s), 
        while {results[most_changes_idx].implementation_name} detected the most changes 
        ({change_pixels[most_changes_idx]:,} pixels).
        """
        
        if accuracy_report and "inter_method_agreement" in accuracy_report:
            agreement = accuracy_report["inter_method_agreement"]
            if "agreement_level" in agreement:
                summary += f" The overall agreement between methods is {agreement['agreement_level'].lower()}."
        
        return summary.strip()
    
    def _create_methods_comparison_table(self, results: List[ChangeDetectionResult]) -> Table:
        """Create methods comparison table."""
        data = [['Method', 'Change Pixels', 'Regions', 'Processing Time (s)', 'Avg Confidence']]
        
        for result in results:
            data.append([
                result.implementation_name,
                f"{result.total_change_pixels:,}",
                str(result.num_change_regions),
                f"{result.processing_time:.3f}",
                f"{result.average_confidence:.3f}" if result.average_confidence > 0 else "N/A"
            ])
        
        table = Table(data, colWidths=[2*inch, 1.2*inch, 0.8*inch, 1.2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightsteelblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table    

    def _create_performance_comparison_table(self, results: List[ChangeDetectionResult]) -> Table:
        """Create performance comparison table with rankings."""
        # Calculate rankings
        processing_times = [(r.processing_time, i) for i, r in enumerate(results)]
        change_pixels = [(r.total_change_pixels, i) for i, r in enumerate(results)]
        
        processing_times.sort()
        change_pixels.sort(reverse=True)
        
        # Create ranking dictionaries
        time_ranks = {idx: rank + 1 for rank, (_, idx) in enumerate(processing_times)}
        change_ranks = {idx: rank + 1 for rank, (_, idx) in enumerate(change_pixels)}
        
        data = [['Method', 'Speed Rank', 'Change Detection Rank', 'Overall Score']]
        
        for i, result in enumerate(results):
            speed_rank = time_ranks[i]
            change_rank = change_ranks[i]
            overall_score = (speed_rank + change_rank) / 2
            
            data.append([
                result.implementation_name,
                f"#{speed_rank}",
                f"#{change_rank}",
                f"{overall_score:.1f}"
            ])
        
        table = Table(data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_accuracy_comparison_table(self, ground_truth_eval: Dict[str, Any]) -> Table:
        """Create accuracy comparison table."""
        data = [['Method', 'Precision', 'Recall', 'F1-Score', 'IoU']]
        
        for method, metrics in ground_truth_eval.items():
            if "error" not in metrics:
                data.append([
                    method,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1_score']:.3f}",
                    f"{metrics['iou']:.3f}"
                ])
        
        table = Table(data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.mistyrose),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_agreement_analysis(self, agreement_data: Dict[str, Any]) -> List:
        """Generate agreement analysis content."""
        content = []
        
        if "error" in agreement_data:
            content.append(Paragraph(f"Agreement analysis error: {agreement_data['error']}", self.styles['Normal']))
            return content
        
        # Overall agreement
        overall = agreement_data.get("overall_agreement", {})
        content.append(Paragraph("Overall Agreement Metrics:", self.styles['SubsectionHeader']))
        
        agreement_text = f"""
        • Mean IoU: {overall.get('mean_iou', 0):.3f}<br/>
        • Mean Jaccard Similarity: {overall.get('mean_jaccard', 0):.3f}<br/>
        • Agreement Level: {agreement_data.get('agreement_level', 'Unknown')}
        """
        content.append(Paragraph(agreement_text, self.styles['MetricStyle']))
        
        # Consensus analysis
        if "consensus_analysis" in agreement_data:
            consensus = agreement_data["consensus_analysis"]
            content.append(Paragraph("Consensus Analysis:", self.styles['SubsectionHeader']))
            
            consensus_text = f"""
            • Full Agreement: {consensus.get('full_agreement_percentage', 0):.1f}% of pixels<br/>
            • Partial Agreement: {consensus.get('partial_agreement_pixels', 0):,} pixels<br/>
            • Consensus Mean: {consensus.get('consensus_mask_mean', 0):.3f}
            """
            content.append(Paragraph(consensus_text, self.styles['MetricStyle']))
        
        return content
    
    def _generate_recommendations(self, results: List[ChangeDetectionResult], 
                                accuracy_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate recommendations based on results."""
        if not results:
            return "No results available for recommendations."
        
        recommendations = []
        
        # Speed recommendation
        processing_times = [r.processing_time for r in results]
        fastest_idx = processing_times.index(min(processing_times))
        recommendations.append(f"For fastest processing, use {results[fastest_idx].implementation_name}.")
        
        # Change detection recommendation
        change_pixels = [r.total_change_pixels for r in results]
        most_sensitive_idx = change_pixels.index(max(change_pixels))
        recommendations.append(f"For maximum sensitivity, use {results[most_sensitive_idx].implementation_name}.")
        
        # Accuracy recommendation (if available)
        if accuracy_report and "ground_truth_evaluation" in accuracy_report:
            gt_eval = accuracy_report["ground_truth_evaluation"]
            best_f1 = 0
            best_method = ""
            
            for method, metrics in gt_eval.items():
                if "error" not in metrics and metrics.get("f1_score", 0) > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_method = method
            
            if best_method:
                recommendations.append(f"For best overall accuracy, use {best_method} (F1-score: {best_f1:.3f}).")
        
        # Agreement recommendation
        if accuracy_report and "inter_method_agreement" in accuracy_report:
            agreement_level = accuracy_report["inter_method_agreement"].get("agreement_level", "")
            if agreement_level == "Low":
                recommendations.append("Consider using multiple methods and ensemble voting due to low agreement.")
            elif agreement_level == "High":
                recommendations.append("High agreement between methods suggests consistent results.")
        
        return " ".join(recommendations)
    
    def _generate_visualizations(self, result: ChangeDetectionResult, 
                               accuracy_metrics: Optional[AccuracyMetrics] = None) -> Optional[str]:
        """Generate visualization plots for individual report using standardized components."""
        try:
            return self.viz_components.create_change_mask_visualization(result)
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            return None
    
    def _generate_comparison_visualizations(self, results: List[ChangeDetectionResult], 
                                          accuracy_report: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generate comparison visualization plots using standardized components."""
        try:
            # Create comprehensive dashboard
            dashboard_path = self.viz_components.create_comprehensive_comparison_dashboard(results)
            
            # Also create individual comparison charts
            performance_path = self.viz_components.create_performance_comparison_chart(results)
            stats_path = self.viz_components.create_change_area_statistics_chart(results)
            
            # If accuracy data is available, create accuracy comparison
            if accuracy_report and "ground_truth_evaluation" in accuracy_report:
                accuracy_data = {}
                for method, metrics in accuracy_report["ground_truth_evaluation"].items():
                    if "error" not in metrics:
                        accuracy_data[method] = AccuracyMetrics(
                            precision=metrics["precision"],
                            recall=metrics["recall"],
                            f1_score=metrics["f1_score"],
                            iou=metrics["iou"],
                            accuracy=metrics["accuracy"],
                            specificity=metrics.get("specificity", 0.0)
                        )
                
                if accuracy_data:
                    accuracy_path = self.viz_components.create_accuracy_comparison_plot(accuracy_data)
            
            return dashboard_path
            
        except Exception as e:
            print(f"Error generating comparison visualizations: {e}")
            return None
    
    def generate_executive_summary_report(self, results: List[ChangeDetectionResult],
                                        accuracy_report: Optional[Dict[str, Any]] = None,
                                        output_path: Optional[str] = None) -> str:
        """
        Generate an executive summary report with high-level findings.
        
        Args:
            results: List of change detection results
            accuracy_report: Optional accuracy evaluation report
            output_path: Optional custom output path
            
        Returns:
            Path to generated PDF report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"executive_summary_{timestamp}.pdf"
            output_path = os.path.join("reports", filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("Infrastructure Change Detection - Executive Summary", self.styles['CustomTitle']))
        story.append(Spacer(1, 30))
        
        # Key Findings
        story.append(Paragraph("Key Findings", self.styles['SectionHeader']))
        key_findings = self._generate_key_findings(results, accuracy_report)
        story.append(Paragraph(key_findings, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Summary Statistics
        story.append(Paragraph("Summary Statistics", self.styles['SectionHeader']))
        summary_table = self._create_executive_summary_table(results)
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Recommendations
        story.append(Paragraph("Strategic Recommendations", self.styles['SectionHeader']))
        strategic_recommendations = self._generate_strategic_recommendations(results, accuracy_report)
        story.append(Paragraph(strategic_recommendations, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Next Steps
        story.append(Paragraph("Next Steps", self.styles['SectionHeader']))
        next_steps = self._generate_next_steps(results, accuracy_report)
        story.append(Paragraph(next_steps, self.styles['Normal']))
        
        # Footer
        footer_text = f"Executive summary generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Spacer(1, 30))
        story.append(Paragraph(footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        print(f"Executive summary report generated: {output_path}")
        return output_path
    
    def _generate_key_findings(self, results: List[ChangeDetectionResult], 
                             accuracy_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate key findings for executive summary."""
        if not results:
            return "No analysis results available."
        
        total_changes = sum(r.total_change_pixels for r in results)
        avg_changes = total_changes / len(results)
        
        findings = f"""
        Analysis of satellite imagery using {len(results)} different detection methods revealed significant 
        infrastructure changes. On average, {avg_changes:,.0f} pixels of change were detected per method, 
        indicating substantial development activity in the analyzed area.
        """
        
        if accuracy_report and "inter_method_agreement" in accuracy_report:
            agreement_level = accuracy_report["inter_method_agreement"].get("agreement_level", "Unknown")
            findings += f" The {agreement_level.lower()} level of agreement between methods provides "
            
            if agreement_level == "High":
                findings += "high confidence in the detected changes."
            elif agreement_level == "Medium":
                findings += "moderate confidence, suggesting some variability in detection approaches."
            else:
                findings += "limited confidence, indicating significant differences in detection methodologies."
        
        return findings.strip()
    
    def _create_executive_summary_table(self, results: List[ChangeDetectionResult]) -> Table:
        """Create executive summary statistics table."""
        if not results:
            return Table([["No data available"]])
        
        total_pixels = sum(r.total_change_pixels for r in results)
        total_regions = sum(r.num_change_regions for r in results)
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        data = [
            ['Metric', 'Value'],
            ['Methods Analyzed', str(len(results))],
            ['Total Change Pixels Detected', f"{total_pixels:,}"],
            ['Total Change Regions', str(total_regions)],
            ['Average Processing Time', f"{avg_processing_time:.3f} seconds"],
            ['Analysis Date', datetime.now().strftime('%Y-%m-%d')]
        ]
        
        table = Table(data, colWidths=[3*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _generate_strategic_recommendations(self, results: List[ChangeDetectionResult], 
                                          accuracy_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate strategic recommendations for executive summary."""
        recommendations = []
        
        if len(results) > 1:
            # Find best performing method
            change_pixels = [r.total_change_pixels for r in results]
            processing_times = [r.processing_time for r in results]
            
            most_sensitive_idx = change_pixels.index(max(change_pixels))
            fastest_idx = processing_times.index(min(processing_times))
            
            recommendations.append(f"For comprehensive change detection, deploy {results[most_sensitive_idx].implementation_name}.")
            recommendations.append(f"For rapid assessment, utilize {results[fastest_idx].implementation_name}.")
        
        if accuracy_report and "ground_truth_evaluation" in accuracy_report:
            # Find most accurate method
            best_f1 = 0
            best_method = ""
            
            for method, metrics in accuracy_report["ground_truth_evaluation"].items():
                if "error" not in metrics and metrics.get("f1_score", 0) > best_f1:
                    best_f1 = metrics["f1_score"]
                    best_method = method
            
            if best_method:
                recommendations.append(f"For highest accuracy, implement {best_method} as the primary detection method.")
        
        recommendations.append("Consider implementing automated monitoring systems for continuous infrastructure surveillance.")
        recommendations.append("Establish regular reporting cycles to track development trends over time.")
        
        return " ".join(recommendations)
    
    def _generate_next_steps(self, results: List[ChangeDetectionResult], 
                           accuracy_report: Optional[Dict[str, Any]] = None) -> str:
        """Generate next steps for executive summary."""
        steps = [
            "1. Review detailed technical reports for each detection method.",
            "2. Validate critical change detections through field verification or higher resolution imagery.",
            "3. Establish baseline metrics for ongoing monitoring programs.",
            "4. Consider integration with GIS systems for spatial analysis and reporting."
        ]
        
        if accuracy_report and "inter_method_agreement" in accuracy_report:
            agreement_level = accuracy_report["inter_method_agreement"].get("agreement_level", "")
            if agreement_level == "Low":
                steps.append("5. Investigate causes of low method agreement and consider ensemble approaches.")
        
        steps.append("6. Plan follow-up analysis with updated satellite imagery to track development progression.")
        
        return " ".join(steps)