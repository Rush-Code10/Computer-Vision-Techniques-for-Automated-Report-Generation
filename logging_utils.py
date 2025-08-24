"""
Logging and progress tracking utilities for the Change Detection System.
Provides structured logging, progress bars, and performance monitoring.
"""

import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional, Any, Dict
from contextlib import contextmanager
from tqdm import tqdm


class ChangeDetectionLogger:
    """Custom logger for the change detection system."""
    
    def __init__(self, name: str = "change_detection", log_level: str = "INFO", 
                 log_to_file: bool = True, log_file: str = "change_detection.log",
                 include_timestamps: bool = True):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_file: Log file path
            include_timestamps: Whether to include timestamps in console output
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        if include_timestamps:
            console_format = '%(asctime)s - %(levelname)s - %(message)s'
        else:
            console_format = '%(levelname)s - %(message)s'
        
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        
        console_formatter = logging.Formatter(console_format, datefmt='%H:%M:%S')
        file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            try:
                # Create log directory if it doesn't exist
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)  # Always log everything to file
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)
                
                self.logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                self.logger.warning(f"Could not set up file logging: {e}")
        
        self.start_time = time.time()
        self.step_times = {}
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, **kwargs)
    
    def log_system_info(self):
        """Log system information."""
        import platform
        import psutil
        
        self.info("=" * 50)
        self.info("SYSTEM INFORMATION")
        self.info("=" * 50)
        self.info(f"Platform: {platform.platform()}")
        self.info(f"Python Version: {platform.python_version()}")
        self.info(f"CPU Count: {psutil.cpu_count()}")
        self.info(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        self.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 50)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration settings."""
        self.info("CONFIGURATION SETTINGS")
        self.info("-" * 30)
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"  {sub_key}: {sub_value}")
            else:
                self.info(f"{key}: {value}")
        self.info("-" * 30)
    
    def start_step(self, step_name: str):
        """Start timing a processing step."""
        self.step_times[step_name] = time.time()
        self.info(f"🚀 Starting: {step_name}")
    
    def end_step(self, step_name: str):
        """End timing a processing step."""
        if step_name in self.step_times:
            elapsed = time.time() - self.step_times[step_name]
            self.info(f"✅ Completed: {step_name} ({elapsed:.3f}s)")
            del self.step_times[step_name]
        else:
            self.info(f"✅ Completed: {step_name}")
    
    def log_results_summary(self, results: list):
        """Log summary of results."""
        self.info("RESULTS SUMMARY")
        self.info("-" * 30)
        for result in results:
            self.info(f"{result.implementation_name}:")
            self.info(f"  Processing Time: {result.processing_time:.3f}s")
            self.info(f"  Change Pixels: {result.total_change_pixels:,}")
            self.info(f"  Change Regions: {result.num_change_regions}")
            self.info(f"  Average Confidence: {result.average_confidence:.3f}")
        self.info("-" * 30)
    
    def log_performance_metrics(self):
        """Log overall performance metrics."""
        total_time = time.time() - self.start_time
        self.info(f"🎯 Total Processing Time: {total_time:.3f} seconds")
        
        # Log memory usage if available
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.info(f"💾 Peak Memory Usage: {memory_mb:.1f} MB")
        except ImportError:
            pass
    
    @contextmanager
    def step_context(self, step_name: str):
        """Context manager for timing steps."""
        self.start_step(step_name)
        try:
            yield
        finally:
            self.end_step(step_name)


class ProgressTracker:
    """Progress tracking utility with optional progress bars."""
    
    def __init__(self, show_progress: bool = True, logger: Optional[ChangeDetectionLogger] = None):
        """
        Initialize progress tracker.
        
        Args:
            show_progress: Whether to show progress bars
            logger: Optional logger instance
        """
        self.show_progress = show_progress
        self.logger = logger
        self.current_bars = {}
    
    def create_progress_bar(self, name: str, total: int, description: str = "") -> Optional[tqdm]:
        """
        Create a progress bar.
        
        Args:
            name: Unique name for the progress bar
            total: Total number of items
            description: Description for the progress bar
            
        Returns:
            tqdm progress bar or None if progress is disabled
        """
        if not self.show_progress:
            if self.logger:
                self.logger.info(f"Starting: {description or name}")
            return None
        
        pbar = tqdm(
            total=total,
            desc=description or name,
            unit="item",
            ncols=80,
            leave=False
        )
        self.current_bars[name] = pbar
        return pbar
    
    def update_progress(self, name: str, increment: int = 1):
        """Update progress bar."""
        if name in self.current_bars and self.current_bars[name]:
            self.current_bars[name].update(increment)
    
    def close_progress_bar(self, name: str):
        """Close and remove progress bar."""
        if name in self.current_bars and self.current_bars[name]:
            self.current_bars[name].close()
            del self.current_bars[name]
            if self.logger:
                self.logger.info(f"Completed: {name}")
    
    def close_all(self):
        """Close all progress bars."""
        for name in list(self.current_bars.keys()):
            self.close_progress_bar(name)
    
    @contextmanager
    def progress_context(self, name: str, total: int, description: str = ""):
        """Context manager for progress tracking."""
        pbar = self.create_progress_bar(name, total, description)
        try:
            yield pbar
        finally:
            self.close_progress_bar(name)


class PerformanceMonitor:
    """Monitor system performance during processing."""
    
    def __init__(self, logger: Optional[ChangeDetectionLogger] = None):
        """Initialize performance monitor."""
        self.logger = logger
        self.start_time = None
        self.checkpoints = {}
        
        try:
            import psutil
            self.psutil_available = True
            self.process = psutil.Process()
            self.initial_memory = self.process.memory_info().rss
        except ImportError:
            self.psutil_available = False
            if logger:
                logger.warning("psutil not available - limited performance monitoring")
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        if self.logger:
            self.logger.info("🔍 Performance monitoring started")
    
    def checkpoint(self, name: str):
        """Create a performance checkpoint."""
        if not self.start_time:
            self.start_monitoring()
        
        checkpoint_data = {
            'time': time.time() - self.start_time,
            'timestamp': datetime.now()
        }
        
        if self.psutil_available:
            memory_info = self.process.memory_info()
            checkpoint_data.update({
                'memory_rss': memory_info.rss,
                'memory_vms': memory_info.vms,
                'memory_delta': memory_info.rss - self.initial_memory,
                'cpu_percent': self.process.cpu_percent()
            })
        
        self.checkpoints[name] = checkpoint_data
        
        if self.logger:
            msg = f"📊 Checkpoint '{name}': {checkpoint_data['time']:.3f}s"
            if self.psutil_available:
                memory_mb = checkpoint_data['memory_rss'] / (1024 * 1024)
                delta_mb = checkpoint_data['memory_delta'] / (1024 * 1024)
                msg += f", Memory: {memory_mb:.1f}MB (+{delta_mb:.1f}MB)"
            self.logger.debug(msg)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.start_time:
            return {}
        
        total_time = time.time() - self.start_time
        summary = {
            'total_time': total_time,
            'checkpoints': len(self.checkpoints),
            'checkpoint_details': self.checkpoints
        }
        
        if self.psutil_available:
            current_memory = self.process.memory_info().rss
            summary.update({
                'initial_memory_mb': self.initial_memory / (1024 * 1024),
                'final_memory_mb': current_memory / (1024 * 1024),
                'memory_delta_mb': (current_memory - self.initial_memory) / (1024 * 1024),
                'peak_memory_mb': max(
                    [cp.get('memory_rss', 0) for cp in self.checkpoints.values()] + [current_memory]
                ) / (1024 * 1024)
            })
        
        return summary
    
    def log_summary(self):
        """Log performance summary."""
        if not self.logger:
            return
        
        summary = self.get_summary()
        if not summary:
            return
        
        self.logger.info("PERFORMANCE SUMMARY")
        self.logger.info("-" * 30)
        self.logger.info(f"Total Time: {summary['total_time']:.3f}s")
        self.logger.info(f"Checkpoints: {summary['checkpoints']}")
        
        if self.psutil_available:
            self.logger.info(f"Memory Usage: {summary['initial_memory_mb']:.1f}MB → {summary['final_memory_mb']:.1f}MB")
            self.logger.info(f"Memory Delta: {summary['memory_delta_mb']:+.1f}MB")
            self.logger.info(f"Peak Memory: {summary['peak_memory_mb']:.1f}MB")
        
        self.logger.info("-" * 30)


def setup_logging(config) -> tuple:
    """
    Set up logging and progress tracking based on configuration.
    
    Args:
        config: System configuration object
        
    Returns:
        Tuple of (logger, progress_tracker, performance_monitor)
    """
    # Create log file path
    log_file_path = config.log_file
    if config.base_directory and not os.path.isabs(log_file_path):
        log_file_path = os.path.join(config.base_directory, log_file_path)
    
    # Initialize logger
    logger = ChangeDetectionLogger(
        name="change_detection",
        log_level=config.log_level,
        log_to_file=config.log_to_file,
        log_file=log_file_path,
        include_timestamps=config.include_timestamps
    )
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker(
        show_progress=config.show_progress,
        logger=logger
    )
    
    # Initialize performance monitor
    performance_monitor = PerformanceMonitor(logger=logger)
    
    return logger, progress_tracker, performance_monitor