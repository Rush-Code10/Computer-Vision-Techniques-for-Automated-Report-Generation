"""
Visualization components for change detection implementations.
Creates consistent visualizations with standardized color schemes and legends.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import cv2

from standardized_data_models import ChangeDetectionResult, AccuracyMetrics


class VisualizationComponents:
    """Provides consistent visualization components for change detection results."""
    
    def __init__(self):
        # Define consistent color scheme
        self.colors = {
            'change': '#FF6B6B',      # Red for changes
            'no_change': '#4ECDC4',   # Teal for no change
            'background': '#F7F7F7',  # Light gray for background
            'confidence_low': '#FFE66D',   # Yellow for low confidence
            'confidence_high': '#FF6B6B',  # Red for high confidence
            'primary': '#2E86AB',     # Blue for primary elements
            'secondary': '#A23B72',   # Purple for secondary elements
            'accent': '#F18F01'       # Orange for accents
        }
        
        # Set consistent style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Define consistent figure parameters
        self.fig_params = {
            'dpi': 150,
            'bbox_inches': 'tight',
            'facecolor': 'white',
            'edgecolor': 'none'
        }
    
    def create_change_mask_visualization(self, result: ChangeDetectionResult, 
                                       original_image: Optional[np.ndarray] = None,
                                       save_path: Optional[str] = None) -> str:
        """
        Create a standardized change mask visualization.
        
        Args:
            result: Change detection result
            original_image: Optional original image for overlay
            save_path: Optional path to save the visualization
            
        Returns:
            Path to saved visualization
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{result.implementation_name} - Change Detection Results', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Change mask
        change_mask_colored = self._apply_change_colormap(result.change_mask)
        axes[0].imshow(change_mask_colored)
        axes[0].set_title('Change Mask', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Add legend for change mask
        legend_elements = [
            patches.Patch(color=self.colors['no_change'], label='No Change'),
            patches.Patch(color=self.colors['change'], label='Change Detected')
        ]
        axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        # Confidence map (if available)
        if result.confidence_map is not None:
            confidence_colored = self._apply_confidence_colormap(result.confidence_map)
            im = axes[1].imshow(confidence_colored)
            axes[1].set_title('Confidence Map', fontsize=12, fontweight='bold')
            axes[1].axis('off')
            
            # Add colorbar for confidence
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Confidence Level', rotation=270, labelpad=15)
        else:
            axes[1].text(0.5, 0.5, 'No confidence map available', 
                        ha='center', va='center', transform=axes[1].transAxes,
                        fontsize=12, color=self.colors['secondary'])
            axes[1].set_title('Confidence Map', fontsize=12, fontweight='bold')
            axes[1].axis('off')
        
        # Overlay visualization
        if original_image is not None:
            overlay = self._create_overlay(original_image, result.change_mask)
            axes[2].imshow(overlay)
            axes[2].set_title('Change Overlay', fontsize=12, fontweight='bold')
        else:
            # Show change regions with bounding boxes
            axes[2].imshow(change_mask_colored)
            self._add_region_boxes(axes[2], result.change_regions)
            axes[2].set_title('Change Regions', fontsize=12, fontweight='bold')
        
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            viz_dir = os.path.join("reports", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, f"{result.implementation_name.lower().replace(' ', '_')}_change_mask.png")
        
        plt.savefig(save_path, **self.fig_params)
        plt.close()
        
        return save_path
    
    def create_accuracy_comparison_plot(self, accuracy_data: Dict[str, AccuracyMetrics],
                                      save_path: Optional[str] = None) -> str:
        """
        Create a comparison plot of accuracy metrics across methods.
        
        Args:
            accuracy_data: Dictionary mapping method names to AccuracyMetrics
            save_path: Optional path to save the plot
            
        Returns:
            Path to saved plot
        """
        methods = list(accuracy_data.keys())
        metrics = ['precision', 'recall', 'f1_score', 'iou', 'accuracy']
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for method in methods:
            acc_metrics = accuracy_data[method]
            data['precision'].append(acc_metrics.precision)
            data['recall'].append(acc_metrics.recall)
            data['f1_score'].append(acc_metrics.f1_score)
            data['iou'].append(acc_metrics.iou)
            data['accuracy'].append(acc_metrics.accuracy)
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Accuracy Metrics Comparison', fontsize=16, fontweight='bold', 
                    color=self.colors['primary'])
        
        # Bar chart
        x = np.arange(len(methods))
        width = 0.15
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent'], 
                 self.colors['change'], self.colors['no_change']]
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i * width, data[metric], width, label=metric.replace('_', ' ').title(),
                   color=colors[i], alpha=0.8)
        
        ax1.set_xlabel('Methods', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Accuracy Metrics by Method', fontweight='bold')
        ax1.set_xticks(x + width * 2)
        ax1.set_xticklabels(methods, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax2 = plt.subplot(122, projection='polar')
        
        for i, method in enumerate(methods):
            values = [data[metric][i] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            ax2.plot(angles, values, 'o-', linewidth=2, label=method, 
                    color=colors[i % len(colors)])
            ax2.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax2.set_ylim(0, 1)
        ax2.set_title('Accuracy Metrics Radar Chart', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            viz_dir = os.path.join("reports", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, "accuracy_comparison.png")
        
        plt.savefig(save_path, **self.fig_params)
        plt.close()
        
        return save_path
    
    def create_performance_comparison_chart(self, results: List[ChangeDetectionResult],
                                          save_path: Optional[str] = None) -> str:
        """
        Create charts showing change area statistics and processing times.
        
        Args:
            results: List of change detection results
            save_path: Optional path to save the chart
            
        Returns:
            Path to saved chart
        """
        methods = [r.implementation_name for r in results]
        processing_times = [r.processing_time for r in results]
        change_pixels = [r.total_change_pixels for r in results]
        num_regions = [r.num_change_regions for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Performance and Change Statistics Comparison', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Processing time comparison
        bars1 = axes[0, 0].bar(methods, processing_times, color=self.colors['primary'], alpha=0.7)
        axes[0, 0].set_title('Processing Time Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars1, processing_times):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{time:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # Change pixels comparison
        bars2 = axes[0, 1].bar(methods, change_pixels, color=self.colors['change'], alpha=0.7)
        axes[0, 1].set_title('Total Change Pixels Detected', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Pixels', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, pixels in zip(bars2, change_pixels):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{pixels:,}', ha='center', va='bottom', fontweight='bold')
        
        # Number of regions comparison
        bars3 = axes[1, 0].bar(methods, num_regions, color=self.colors['secondary'], alpha=0.7)
        axes[1, 0].set_title('Number of Change Regions', fontweight='bold')
        axes[1, 0].set_ylabel('Number of Regions', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, regions in zip(bars3, num_regions):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{regions}', ha='center', va='bottom', fontweight='bold')
        
        # Combined efficiency plot (change detection vs processing time)
        scatter = axes[1, 1].scatter(processing_times, change_pixels, 
                                   s=100, c=range(len(methods)), 
                                   cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('Processing Time (seconds)', fontweight='bold')
        axes[1, 1].set_ylabel('Change Pixels Detected', fontweight='bold')
        axes[1, 1].set_title('Efficiency Analysis', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add method labels to scatter points
        for i, method in enumerate(methods):
            axes[1, 1].annotate(method, (processing_times[i], change_pixels[i]),
                              xytext=(5, 5), textcoords='offset points',
                              fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            viz_dir = os.path.join("reports", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, "performance_comparison.png")
        
        plt.savefig(save_path, **self.fig_params)
        plt.close()
        
        return save_path
    
    def create_change_area_statistics_chart(self, results: List[ChangeDetectionResult],
                                          save_path: Optional[str] = None) -> str:
        """
        Create detailed charts showing change area statistics.
        
        Args:
            results: List of change detection results
            save_path: Optional path to save the chart
            
        Returns:
            Path to saved chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Change Area Statistics Analysis', 
                    fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        methods = [r.implementation_name for r in results]
        
        # Change area percentage by method
        change_percentages = []
        for result in results:
            total_pixels = result.image_dimensions[0] * result.image_dimensions[1]
            percentage = (result.total_change_pixels / total_pixels) * 100
            change_percentages.append(percentage)
        
        bars1 = axes[0, 0].bar(methods, change_percentages, color=self.colors['accent'], alpha=0.7)
        axes[0, 0].set_title('Change Area Percentage', fontweight='bold')
        axes[0, 0].set_ylabel('Percentage (%)', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, pct in zip(bars1, change_percentages):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # Region size distribution (if regions available)
        region_data_available = any(len(r.change_regions) > 0 for r in results)
        
        if region_data_available:
            # Box plot of region sizes
            region_sizes_by_method = []
            method_labels = []
            
            for result in results:
                if result.change_regions:
                    sizes = [r.area_pixels for r in result.change_regions]
                    region_sizes_by_method.append(sizes)
                    method_labels.append(result.implementation_name)
            
            if region_sizes_by_method:
                axes[0, 1].boxplot(region_sizes_by_method, labels=method_labels)
                axes[0, 1].set_title('Region Size Distribution', fontweight='bold')
                axes[0, 1].set_ylabel('Region Size (pixels)', fontweight='bold')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No region data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes,
                           fontsize=12, color=self.colors['secondary'])
            axes[0, 1].set_title('Region Size Distribution', fontweight='bold')
        
        # Average region size comparison
        avg_region_sizes = []
        for result in results:
            if result.change_regions:
                sizes = [r.area_pixels for r in result.change_regions]
                avg_size = np.mean(sizes) if sizes else 0
            else:
                avg_size = result.total_change_pixels / max(result.num_change_regions, 1)
            avg_region_sizes.append(avg_size)
        
        bars3 = axes[1, 0].bar(methods, avg_region_sizes, color=self.colors['no_change'], alpha=0.7)
        axes[1, 0].set_title('Average Region Size', fontweight='bold')
        axes[1, 0].set_ylabel('Average Size (pixels)', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars3, avg_region_sizes):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{size:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # Change detection efficiency (pixels per second)
        efficiency = [r.total_change_pixels / r.processing_time for r in results]
        bars4 = axes[1, 1].bar(methods, efficiency, color=self.colors['primary'], alpha=0.7)
        axes[1, 1].set_title('Detection Efficiency', fontweight='bold')
        axes[1, 1].set_ylabel('Pixels Detected per Second', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, eff in zip(bars4, efficiency):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{eff:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        if save_path is None:
            viz_dir = os.path.join("reports", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, "change_area_statistics.png")
        
        plt.savefig(save_path, **self.fig_params)
        plt.close()
        
        return save_path
    
    def create_comprehensive_comparison_dashboard(self, results: List[ChangeDetectionResult],
                                                accuracy_data: Optional[Dict[str, AccuracyMetrics]] = None,
                                                save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive dashboard with all comparison visualizations.
        
        Args:
            results: List of change detection results
            accuracy_data: Optional accuracy metrics for each method
            save_path: Optional path to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle('Change Detection Methods - Comprehensive Comparison Dashboard', 
                    fontsize=20, fontweight='bold', color=self.colors['primary'])
        
        methods = [r.implementation_name for r in results]
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Processing time comparison
        ax1 = fig.add_subplot(gs[0, 0])
        processing_times = [r.processing_time for r in results]
        bars1 = ax1.bar(methods, processing_times, color=self.colors['primary'], alpha=0.7)
        ax1.set_title('Processing Time', fontweight='bold')
        ax1.set_ylabel('Seconds')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Change pixels detected
        ax2 = fig.add_subplot(gs[0, 1])
        change_pixels = [r.total_change_pixels for r in results]
        bars2 = ax2.bar(methods, change_pixels, color=self.colors['change'], alpha=0.7)
        ax2.set_title('Change Pixels Detected', fontweight='bold')
        ax2.set_ylabel('Pixels')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Number of regions
        ax3 = fig.add_subplot(gs[0, 2])
        num_regions = [r.num_change_regions for r in results]
        bars3 = ax3.bar(methods, num_regions, color=self.colors['secondary'], alpha=0.7)
        ax3.set_title('Number of Regions', fontweight='bold')
        ax3.set_ylabel('Regions')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Efficiency scatter plot
        ax4 = fig.add_subplot(gs[0, 3])
        efficiency = [r.total_change_pixels / r.processing_time for r in results]
        scatter = ax4.scatter(processing_times, change_pixels, s=100, 
                            c=range(len(methods)), cmap='viridis', alpha=0.7)
        ax4.set_xlabel('Processing Time (s)')
        ax4.set_ylabel('Change Pixels')
        ax4.set_title('Efficiency Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add method labels
        for i, method in enumerate(methods):
            ax4.annotate(method[:10], (processing_times[i], change_pixels[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 5. Accuracy comparison (if available)
        if accuracy_data:
            ax5 = fig.add_subplot(gs[1, :2])
            metrics = ['precision', 'recall', 'f1_score', 'iou']
            x = np.arange(len(methods))
            width = 0.2
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['change']]
            
            for i, metric in enumerate(metrics):
                values = [getattr(accuracy_data[method], metric) for method in methods]
                ax5.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
                       color=colors[i], alpha=0.8)
            
            ax5.set_xlabel('Methods')
            ax5.set_ylabel('Score')
            ax5.set_title('Accuracy Metrics Comparison', fontweight='bold')
            ax5.set_xticks(x + width * 1.5)
            ax5.set_xticklabels(methods, rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim(0, 1.1)
            
            # 6. Radar chart for accuracy
            ax6 = fig.add_subplot(gs[1, 2:], projection='polar')
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]
            
            for i, method in enumerate(methods):
                values = [getattr(accuracy_data[method], metric) for metric in metrics]
                values += values[:1]
                
                ax6.plot(angles, values, 'o-', linewidth=2, label=method,
                        color=colors[i % len(colors)])
                ax6.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
            
            ax6.set_xticks(angles[:-1])
            ax6.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
            ax6.set_ylim(0, 1)
            ax6.set_title('Accuracy Radar Chart', fontweight='bold', pad=20)
            ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 7. Change area statistics
        ax7 = fig.add_subplot(gs[2, :2])
        change_percentages = []
        for result in results:
            total_pixels = result.image_dimensions[0] * result.image_dimensions[1]
            percentage = (result.total_change_pixels / total_pixels) * 100
            change_percentages.append(percentage)
        
        bars7 = ax7.bar(methods, change_percentages, color=self.colors['accent'], alpha=0.7)
        ax7.set_title('Change Area Percentage', fontweight='bold')
        ax7.set_ylabel('Percentage (%)')
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, pct in zip(bars7, change_percentages):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{pct:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 8. Summary statistics table
        ax8 = fig.add_subplot(gs[2, 2:])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create summary table data
        table_data = []
        headers = ['Method', 'Time (s)', 'Pixels', 'Regions', 'Area (%)', 'Efficiency']
        
        for i, result in enumerate(results):
            total_pixels = result.image_dimensions[0] * result.image_dimensions[1]
            area_pct = (result.total_change_pixels / total_pixels) * 100
            eff = result.total_change_pixels / result.processing_time
            
            row = [
                result.implementation_name[:15],
                f'{result.processing_time:.3f}',
                f'{result.total_change_pixels:,}',
                str(result.num_change_regions),
                f'{area_pct:.2f}%',
                f'{eff:.0f}'
            ]
            table_data.append(row)
        
        table = ax8.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor(self.colors['primary'])
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax8.set_title('Summary Statistics', fontweight='bold')
        
        # Save dashboard
        if save_path is None:
            viz_dir = os.path.join("reports", "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            save_path = os.path.join(viz_dir, "comprehensive_dashboard.png")
        
        plt.savefig(save_path, **self.fig_params)
        plt.close()
        
        return save_path
    
    def _apply_change_colormap(self, change_mask: np.ndarray) -> np.ndarray:
        """Apply consistent colormap to change mask."""
        # Create RGB image
        colored_mask = np.zeros((*change_mask.shape, 3), dtype=np.uint8)
        
        # No change areas (background)
        no_change = change_mask == 0
        colored_mask[no_change] = [int(self.colors['background'][1:3], 16),
                                  int(self.colors['background'][3:5], 16),
                                  int(self.colors['background'][5:7], 16)]
        
        # Change areas
        change = change_mask > 0
        colored_mask[change] = [int(self.colors['change'][1:3], 16),
                               int(self.colors['change'][3:5], 16),
                               int(self.colors['change'][5:7], 16)]
        
        return colored_mask
    
    def _apply_confidence_colormap(self, confidence_map: np.ndarray) -> np.ndarray:
        """Apply consistent colormap to confidence map."""
        # Normalize confidence map to 0-1 range
        conf_normalized = (confidence_map - confidence_map.min()) / (confidence_map.max() - confidence_map.min())
        
        # Apply colormap
        cmap = plt.cm.get_cmap('YlOrRd')  # Yellow to Red colormap
        colored_conf = cmap(conf_normalized)
        
        return colored_conf
    
    def _create_overlay(self, original_image: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
        """Create overlay of change mask on original image."""
        if len(original_image.shape) == 3:
            overlay = original_image.copy()
        else:
            overlay = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # Create change overlay
        change_areas = change_mask > 0
        overlay[change_areas] = [255, 107, 107]  # Red overlay for changes
        
        return overlay
    
    def _add_region_boxes(self, ax, change_regions: List) -> None:
        """Add bounding boxes for change regions."""
        for i, region in enumerate(change_regions[:10]):  # Limit to first 10 regions
            x, y, w, h = region.bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=self.colors['accent'],
                           facecolor='none', alpha=0.8)
            ax.add_patch(rect)
            
            # Add region ID label
            ax.text(x, y-5, f'R{region.id}', fontsize=8, fontweight='bold',
                   color=self.colors['accent'], bbox=dict(boxstyle="round,pad=0.3",
                   facecolor='white', alpha=0.8))