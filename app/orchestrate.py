"""
Main Orchestrator for Data Pipeline

This module orchestrates the execution of all pipeline stages in sequence.
"""

import sys
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import configuration and utilities directly to avoid dependency issues
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
import config
PipelineConfig = config.PipelineConfig

# Simple implementations for missing utilities
class Logger:
    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
    
    def info(self, message: str):
        print(f"[INFO] {self.name}: {message}")
    
    def warning(self, message: str):
        print(f"[WARNING] {self.name}: {message}")
    
    def error(self, message: str):
        print(f"[ERROR] {self.name}: {message}")
    
    def debug(self, message: str):
        print(f"[DEBUG] {self.name}: {message}")

class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
    
    def start(self):
        pass
    
    def stop(self):
        return {}

def validate_config(config):
    return []
# Import real zone processors
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'zones'))

# Import each zone processor
from temporal_landing import TemporalLandingProcessor
from persistent_landing import PersistentLandingProcessor
from formatted_documents import FormattedDocumentsProcessor
from formatted_images import FormattedImagesProcessor
from trusted_images import TrustedImagesProcessor
from trusted_documents import TrustedDocumentsProcessor
from exploitation_documents import ExploitationDocumentsProcessor


class PipelineOrchestrator:
    """Main orchestrator for the data pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize the orchestrator."""
        self.config = config
        self.logger = Logger("orchestrator", config.get("monitoring.log_level", "INFO"))
        self.monitor = PerformanceMonitor(config)
        
        # Pipeline stages
        self.stages = [
            ("temporal_landing", TemporalLandingProcessor),
            ("persistent_landing", PersistentLandingProcessor),
            ("formatted_documents", FormattedDocumentsProcessor),
            ("formatted_images", FormattedImagesProcessor),
            ("trusted_images", TrustedImagesProcessor),
            ("trusted_documents", TrustedDocumentsProcessor),
            ("exploitation_documents", ExploitationDocumentsProcessor),
        ]
        
        # Results storage
        self.results: Dict[str, Any] = {}
        self.errors: List[Dict[str, Any]] = []
    
    def validate_environment(self) -> List[str]:
        """Validate the environment and configuration."""
        issues = validate_config(self.config)
        
        if issues:
            self.logger.error("Configuration validation failed:")
            for issue in issues:
                self.logger.error(f"  - {issue}")
        
        return issues
    
    def run_stage(self, stage_name: str, processor_class) -> Dict[str, Any]:
        """Run a single pipeline stage."""
        self.logger.info(f"[STARTING] Starting stage: {stage_name}")
        
        stage_start_time = time.time()
        
        try:
            processor = processor_class(self.config)
            result = processor.process()
            
            stage_duration = time.time() - stage_start_time
            result["stage_duration"] = stage_duration
            
            self.logger.info(f"[SUCCESS] Stage {stage_name} completed successfully in {stage_duration:.2f}s")
            return result
        
        except Exception as e:
            stage_duration = time.time() - stage_start_time
            error_info = {
                "stage": stage_name,
                "error": str(e),
                "duration": stage_duration,
                "timestamp": time.time()
            }
            self.errors.append(error_info)
            
            self.logger.error(f"[ERROR] Stage {stage_name} failed after {stage_duration:.2f}s: {e}")
            raise
    
    def run_pipeline(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete pipeline or specified stages."""
        import time
        pipeline_start_time = time.time()
        
        # Validate environment
        issues = self.validate_environment()
        if issues:
            raise RuntimeError(f"Environment validation failed: {issues}")
        
        # Determine which stages to run
        stages_to_run = stages or [stage[0] for stage in self.stages]
        
        self.logger.info(f"[STARTING] Starting pipeline with stages: {stages_to_run}")
        
        try:
            for stage_name, processor_class in self.stages:
                if stage_name in stages_to_run:
                    result = self.run_stage(stage_name, processor_class)
                    self.results[stage_name] = result
                else:
                    self.logger.info(f"[SKIP] Skipping stage: {stage_name}")
            
            # Calculate total execution time
            pipeline_end_time = time.time()
            total_execution_time = pipeline_end_time - pipeline_start_time
            
            # Generate final report
            final_metrics = {
                "stages_completed": len(self.results),
                "stages_failed": len(self.errors),
                "total_stages": len(stages_to_run),
                "execution_time": total_execution_time,
                "results": self.results,
                "errors": self.errors
            }
            
            self.logger.info(f"[SUCCESS] Pipeline completed successfully!")
            self.logger.info(f"[METRICS] Final metrics: {final_metrics}")
            
            return final_metrics
        
        except Exception as e:
            final_metrics = self.monitor.stop()
            final_metrics.update({
                "stages_completed": len(self.results),
                "stages_failed": len(self.errors),
                "total_stages": len(stages_to_run),
                "results": self.results,
                "errors": self.errors,
                "pipeline_error": str(e)
            })
            
            self.logger.error(f"[ERROR] Pipeline failed: {e}")
            return final_metrics
    
    def run_single_stage(self, stage_name: str) -> Dict[str, Any]:
        """Run a single pipeline stage."""
        stage_map = {stage[0]: stage[1] for stage in self.stages}
        
        if stage_name not in stage_map:
            available_stages = list(stage_map.keys())
            raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {available_stages}")
        
        return self.run_pipeline([stage_name])
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            "stages_available": [stage[0] for stage in self.stages],
            "stages_completed": list(self.results.keys()),
            "stages_failed": [error["stage"] for error in self.errors],
            "total_results": len(self.results),
            "total_errors": len(self.errors)
        }


def main():
    """Main entry point for pipeline orchestration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Pipeline Orchestrator")
    parser.add_argument("--config", default="app/config/pipeline.yaml", help="Configuration file path")
    parser.add_argument("--stages", nargs="+", help="Specific stages to run")
    parser.add_argument("--stage", help="Single stage to run")
    parser.add_argument("--dry-run", action="store_true", help="Run in dry-run mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Load configuration
    config = PipelineConfig(args.config)
    
    # Override configuration with command line arguments
    if args.dry_run:
        config._config["pipeline"]["dry_run"] = True
    
    if args.verbose:
        config._config["monitoring"]["log_level"] = "DEBUG"
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config)
    
    try:
        if args.stage:
            # Run single stage
            result = orchestrator.run_single_stage(args.stage)
        else:
            # Run pipeline
            result = orchestrator.run_pipeline(args.stages)
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Stages completed: {result.get('stages_completed', 0)}")
        print(f"Stages failed: {result.get('stages_failed', 0)}")
        print(f"Total execution time: {result.get('execution_time', 0):.2f}s")
        
        if result.get('errors'):
            print("\nErrors:")
            for error in result['errors']:
                print(f"  - {error['stage']}: {error['error']}")
        
        print("="*60)
        
        # Exit with appropriate code
        if result.get('stages_failed', 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
