"""
Basic monitoring module for the data pipeline.

This module provides monitoring capabilities for pipeline execution,
resource usage, and performance metrics.
"""

import time
import json
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import configuration and shared utilities
from .config import PipelineConfig
from .shared import Logger, utc_timestamp


def get_system_info() -> Dict[str, Any]:
    """Get basic system information without external dependencies."""
    try:
        import psutil
        return {
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used": psutil.virtual_memory().used,
            "memory_available": psutil.virtual_memory().available,
            "memory_total": psutil.virtual_memory().total,
            "disk_percent": psutil.disk_usage('.').percent,
            "disk_used": psutil.disk_usage('.').used,
            "disk_free": psutil.disk_usage('.').free,
            "disk_total": psutil.disk_usage('.').total,
            "cpu_percent": psutil.cpu_percent(),
            "cpu_count": psutil.cpu_count(),
            "psutil_available": True
        }
    except ImportError:
        # Fallback without psutil
        return {
            "memory_percent": 0,
            "memory_used": 0,
            "memory_available": 0,
            "memory_total": 0,
            "disk_percent": 0,
            "disk_used": 0,
            "disk_free": 0,
            "disk_total": 0,
            "cpu_percent": 0,
            "cpu_count": os.cpu_count() or 1,
            "psutil_available": False
        }


class PipelineMonitor:
    """Monitor for pipeline execution and resource usage."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the monitor."""
        self.config = config
        self.logger = Logger("monitor", config.get("monitoring.log_level", "INFO"))
        self.enabled = config.get("monitoring.enabled", True)
        
        # Monitoring data
        self.metrics: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        self.stage_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Resource monitoring
        initial_info = get_system_info()
        self.initial_memory = initial_info["memory_used"]
        self.initial_disk = initial_info["disk_used"]
        self.peak_memory = self.initial_memory
        
        # Performance tracking
        self.performance_data: List[Dict[str, Any]] = []
    
    def start_monitoring(self):
        """Start monitoring."""
        if not self.enabled:
            return
        
        self.start_time = time.time()
        self.logger.info("Pipeline monitoring started")
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return final metrics."""
        if not self.enabled or self.start_time is None:
            return {}
        
        end_time = time.time()
        execution_time = end_time - self.start_time
        
        # Get final resource usage
        final_info = get_system_info()
        final_memory = final_info["memory_used"]
        final_disk = final_info["disk_used"]
        
        # Calculate metrics
        metrics = {
            "execution_time": execution_time,
            "memory_usage": final_memory - self.initial_memory,
            "disk_usage": final_disk - self.initial_disk,
            "peak_memory": self.peak_memory,
            "current_memory_percent": final_info["memory_percent"],
            "current_disk_percent": final_info["disk_percent"],
            "cpu_percent": final_info["cpu_percent"],
            "stage_count": len(self.stage_metrics),
            "timestamp": utc_timestamp()
        }
        
        self.metrics.update(metrics)
        self.logger.info(f"Pipeline monitoring stopped. Execution time: {execution_time:.2f}s")
        
        return metrics
    
    def record_stage_start(self, stage_name: str):
        """Record the start of a pipeline stage."""
        if not self.enabled:
            return
        
        current_info = get_system_info()
        self.stage_metrics[stage_name] = {
            "start_time": time.time(),
            "start_memory": current_info["memory_used"],
            "start_disk": current_info["disk_used"]
        }
        
        self.logger.info(f"Stage monitoring started: {stage_name}")
    
    def record_stage_end(self, stage_name: str, result: Optional[Dict[str, Any]] = None):
        """Record the end of a pipeline stage."""
        if not self.enabled or stage_name not in self.stage_metrics:
            return
        
        end_time = time.time()
        stage_data = self.stage_metrics[stage_name]
        
        # Calculate stage metrics
        stage_duration = end_time - stage_data["start_time"]
        current_info = get_system_info()
        stage_memory = current_info["memory_used"] - stage_data["start_memory"]
        stage_disk = current_info["disk_used"] - stage_data["start_disk"]
        
        # Update peak memory
        if current_info["memory_used"] > self.peak_memory:
            self.peak_memory = current_info["memory_used"]
        
        # Store stage metrics
        stage_metrics = {
            "duration": stage_duration,
            "memory_usage": stage_memory,
            "disk_usage": stage_disk,
            "end_time": end_time,
            "result": result or {}
        }
        
        self.stage_metrics[stage_name].update(stage_metrics)
        
        # Add to performance data
        self.performance_data.append({
            "stage": stage_name,
            "timestamp": utc_timestamp(),
            "duration": stage_duration,
            "memory_usage": stage_memory,
            "disk_usage": stage_disk,
            "success": result is not None
        })
        
        self.logger.info(f"Stage monitoring ended: {stage_name} (duration: {stage_duration:.2f}s)")
    
    def record_error(self, stage_name: str, error: str):
        """Record an error for a pipeline stage."""
        if not self.enabled:
            return
        
        current_info = get_system_info()
        error_data = {
            "stage": stage_name,
            "error": error,
            "timestamp": utc_timestamp(),
            "memory_at_error": current_info["memory_used"],
            "disk_at_error": current_info["disk_used"]
        }
        
        if "errors" not in self.metrics:
            self.metrics["errors"] = []
        
        self.metrics["errors"].append(error_data)
        self.logger.error(f"Error recorded for stage {stage_name}: {error}")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.enabled:
            return {}
        
        current_info = get_system_info()
        return {
            "memory_percent": current_info["memory_percent"],
            "memory_used": current_info["memory_used"],
            "memory_available": current_info["memory_available"],
            "disk_percent": current_info["disk_percent"],
            "disk_used": current_info["disk_used"],
            "disk_free": current_info["disk_free"],
            "cpu_percent": current_info["cpu_percent"],
            "timestamp": utc_timestamp()
        }
    
    def save_metrics(self, filepath: Optional[str] = None) -> str:
        """Save metrics to a JSON file."""
        if not self.enabled:
            return ""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"pipeline_metrics_{timestamp}.json"
        
        # Prepare metrics data
        system_info = get_system_info()
        metrics_data = {
            "pipeline_metrics": self.metrics,
            "stage_metrics": self.stage_metrics,
            "performance_data": self.performance_data,
            "system_info": {
                "cpu_count": system_info["cpu_count"],
                "memory_total": system_info["memory_total"],
                "disk_total": system_info["disk_total"],
                "platform": os.name,
                "python_version": os.sys.version,
                "psutil_available": system_info["psutil_available"]
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics saved to: {filepath}")
        return filepath
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring data."""
        if not self.enabled:
            return {}
        
        total_duration = 0
        total_memory = 0
        total_disk = 0
        successful_stages = 0
        failed_stages = 0
        
        for stage_name, stage_data in self.stage_metrics.items():
            if "duration" in stage_data:
                total_duration += stage_data["duration"]
                total_memory += stage_data.get("memory_usage", 0)
                total_disk += stage_data.get("disk_usage", 0)
                
                if stage_data.get("success", True):
                    successful_stages += 1
                else:
                    failed_stages += 1
        
        current_info = get_system_info()
        return {
            "total_duration": total_duration,
            "total_memory_usage": total_memory,
            "total_disk_usage": total_disk,
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "total_stages": len(self.stage_metrics),
            "peak_memory": self.peak_memory,
            "current_memory_percent": current_info["memory_percent"],
            "current_disk_percent": current_info["disk_percent"],
            "error_count": len(self.metrics.get("errors", []))
        }


class ResourceMonitor:
    """Monitor system resources during pipeline execution."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the resource monitor."""
        self.config = config
        self.logger = Logger("resource_monitor", config.get("monitoring.log_level", "INFO"))
        self.enabled = config.get("monitoring.resource_monitoring", True)
        self.monitoring_interval = 5  # seconds
        self.monitoring_data: List[Dict[str, Any]] = []
        self.monitoring_active = False
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self.enabled:
            return
        
        self.monitoring_active = True
        self.monitoring_data = []
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.enabled:
            return
        
        self.monitoring_active = False
        self.logger.info("Resource monitoring stopped")
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics."""
        if not self.enabled:
            return {}
        
        current_info = get_system_info()
        metrics = {
            "timestamp": utc_timestamp(),
            "memory": {
                "percent": current_info["memory_percent"],
                "used": current_info["memory_used"],
                "available": current_info["memory_available"],
                "total": current_info["memory_total"]
            },
            "disk": {
                "percent": current_info["disk_percent"],
                "used": current_info["disk_used"],
                "free": current_info["disk_free"],
                "total": current_info["disk_total"]
            },
            "cpu": {
                "percent": current_info["cpu_percent"],
                "count": current_info["cpu_count"]
            }
        }
        
        if self.monitoring_active:
            self.monitoring_data.append(metrics)
        
        return metrics
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring data."""
        if not self.enabled or not self.monitoring_data:
            return {}
        
        memory_percents = [m["memory"]["percent"] for m in self.monitoring_data]
        disk_percents = [m["disk"]["percent"] for m in self.monitoring_data]
        cpu_percents = [m["cpu"]["percent"] for m in self.monitoring_data]
        
        return {
            "monitoring_duration": len(self.monitoring_data) * self.monitoring_interval,
            "memory": {
                "avg_percent": sum(memory_percents) / len(memory_percents),
                "max_percent": max(memory_percents),
                "min_percent": min(memory_percents)
            },
            "disk": {
                "avg_percent": sum(disk_percents) / len(disk_percents),
                "max_percent": max(disk_percents),
                "min_percent": min(disk_percents)
            },
            "cpu": {
                "avg_percent": sum(cpu_percents) / len(cpu_percents),
                "max_percent": max(cpu_percents),
                "min_percent": min(cpu_percents)
            },
            "data_points": len(self.monitoring_data)
        }


def create_monitoring_report(monitor: PipelineMonitor, resource_monitor: ResourceMonitor) -> str:
    """Create a comprehensive monitoring report."""
    pipeline_summary = monitor.get_summary()
    resource_summary = resource_monitor.get_monitoring_summary()
    
    report = f"""
# Pipeline Monitoring Report

## Pipeline Summary
- Total Duration: {pipeline_summary.get('total_duration', 0):.2f} seconds
- Successful Stages: {pipeline_summary.get('successful_stages', 0)}
- Failed Stages: {pipeline_summary.get('failed_stages', 0)}
- Total Stages: {pipeline_summary.get('total_stages', 0)}
- Peak Memory Usage: {pipeline_summary.get('peak_memory', 0) / (1024**3):.2f} GB
- Current Memory Usage: {pipeline_summary.get('current_memory_percent', 0):.1f}%
- Current Disk Usage: {pipeline_summary.get('current_disk_percent', 0):.1f}%
- Error Count: {pipeline_summary.get('error_count', 0)}

## Resource Monitoring Summary
- Monitoring Duration: {resource_summary.get('monitoring_duration', 0)} seconds
- Average Memory Usage: {resource_summary.get('memory', {}).get('avg_percent', 0):.1f}%
- Maximum Memory Usage: {resource_summary.get('memory', {}).get('max_percent', 0):.1f}%
- Average CPU Usage: {resource_summary.get('cpu', {}).get('avg_percent', 0):.1f}%
- Maximum CPU Usage: {resource_summary.get('cpu', {}).get('max_percent', 0):.1f}%
- Data Points Collected: {resource_summary.get('data_points', 0)}

## Stage Details
"""
    
    for stage_name, stage_data in monitor.stage_metrics.items():
        duration = stage_data.get('duration', 0)
        memory = stage_data.get('memory_usage', 0)
        success = stage_data.get('success', True)
        status = "✅ Success" if success else "❌ Failed"
        
        report += f"- **{stage_name}**: {duration:.2f}s, {memory / (1024**2):.1f} MB, {status}\n"
    
    return report


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig()
    monitor = PipelineMonitor(config)
    resource_monitor = ResourceMonitor(config)
    
    # Start monitoring
    monitor.start_monitoring()
    resource_monitor.start_monitoring()
    
    # Simulate some work
    time.sleep(2)
    
    # Record a stage
    monitor.record_stage_start("test_stage")
    time.sleep(1)
    monitor.record_stage_end("test_stage", {"processed_records": 100})
    
    # Stop monitoring
    monitor.stop_monitoring()
    resource_monitor.stop_monitoring()
    
    # Generate report
    report = create_monitoring_report(monitor, resource_monitor)
    print(report)
    
    # Save metrics
    metrics_file = monitor.save_metrics()
    print(f"Metrics saved to: {metrics_file}")