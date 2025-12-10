#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging and Performance Tracking Module
Provides structured logging and timing capabilities
"""

import logging
import time
from pathlib import Path
from typing import Optional, TextIO
from contextlib import contextmanager


class PipelineLogger:
    """
    Unified logger for the donations pipeline
    Combines application logging with performance tracking
    """
    
    def __init__(self, name: str = "donations_pipeline", log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Timing entries
        self.timing_entries: list[tuple[str, float]] = []
    
    @contextmanager
    def timed_operation(self, operation_name: str):
        """
        Context manager for timing operations
        
        Usage:
            with logger.timed_operation("Data Loading"):
                # your code here
        """
        self.logger.info(f"Starting: {operation_name}")
        start_time = time.perf_counter()
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Failed: {operation_name} - {str(e)}", exc_info=True)
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            self.timing_entries.append((operation_name, elapsed))
            self.logger.info(f"Completed: {operation_name} ({elapsed:.2f}s)")
    
    def save_timing_report(self, output_path: Path) -> None:
        """Save timing report to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Operation Timing Report\n")
            f.write("=" * 80 + "\n\n")
            
            total_time = sum(t for _, t in self.timing_entries)
            
            for operation, duration in self.timing_entries:
                percentage = (duration / total_time * 100) if total_time > 0 else 0
                f.write(f"{operation:<50} {duration:>10.2f}s ({percentage:>5.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{'Total Time':<50} {total_time:>10.2f}s\n")
        
        self.logger.info(f"Timing report saved to: {output_path}")
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message"""
        self.logger.error(message, exc_info=exc_info)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)


def setup_logger(output_dir: Path, name: str = "donations_pipeline") -> PipelineLogger:
    """
    Factory function to create a configured logger
    
    Args:
        output_dir: Directory to store log files
        name: Logger name
    
    Returns:
        Configured PipelineLogger instance
    """
    log_file = output_dir / "pipeline.log"
    return PipelineLogger(name=name, log_file=log_file)