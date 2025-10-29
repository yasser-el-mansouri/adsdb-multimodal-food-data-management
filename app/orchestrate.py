"""
Main Orchestrator for Data Pipeline

This module orchestrates the execution of all pipeline stages in sequence.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the parent directory to sys.path so we can import 'app' modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import configuration and utilities
from app.utils.config import PipelineConfig, validate_config
from app.utils.monitoring import PipelineMonitor, ResourceMonitor, create_monitoring_report
from app.utils.shared import Logger
from app.zones.exploitation_zone.exploitation_documents import ExploitationDocumentsProcessor
from app.zones.exploitation_zone.exploitation_images import ExploitationImagesProcessor
from app.zones.formatted_zone.formatted_documents import FormattedDocumentsProcessor
from app.zones.formatted_zone.formatted_images import FormattedImagesProcessor
from app.zones.landing_zone.persistent_landing import PersistentLandingProcessor

# Import each zone processor from their respective zone folders
from app.zones.landing_zone.temporal_landing import TemporalLandingProcessor
from app.zones.trusted_zone.trusted_documents import TrustedDocumentsProcessor
from app.zones.trusted_zone.trusted_images import TrustedImagesProcessor


class PipelineOrchestrator:
    """Main orchestrator for the data pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize the orchestrator."""
        self.config = config
        self.logger = Logger("orchestrator", config.get("monitoring.log_level", "INFO"))
        self.monitor = PipelineMonitor(config)
        self.resource_monitor = ResourceMonitor(config)

        # Pipeline stages
        self.stages = [
            ("temporal_landing", TemporalLandingProcessor),
            ("persistent_landing", PersistentLandingProcessor),
            ("formatted_documents", FormattedDocumentsProcessor),
            ("formatted_images", FormattedImagesProcessor),
            ("trusted_images", TrustedImagesProcessor),
            ("trusted_documents", TrustedDocumentsProcessor),
            ("exploitation_documents", ExploitationDocumentsProcessor),
            ("exploitation_images", ExploitationImagesProcessor),
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

        # Start stage monitoring
        self.monitor.record_stage_start(stage_name)
        stage_start_time = time.time()

        try:
            processor = processor_class(self.config)
            result = processor.process()

            stage_duration = time.time() - stage_start_time
            result["stage_duration"] = stage_duration

            # Record successful stage end
            self.monitor.record_stage_end(stage_name, result)

            self.logger.info(
                f"[SUCCESS] Stage {stage_name} completed successfully in {stage_duration:.2f}s"
            )
            return result

        except Exception as e:
            stage_duration = time.time() - stage_start_time
            error_info = {
                "stage": stage_name,
                "error": str(e),
                "duration": stage_duration,
                "timestamp": time.time(),
            }
            self.errors.append(error_info)

            # Record error
            self.monitor.record_error(stage_name, str(e))

            self.logger.error(f"[ERROR] Stage {stage_name} failed after {stage_duration:.2f}s: {e}")
            raise

    def run_pipeline(self, stages: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run the complete pipeline or specified stages."""
        import time

        pipeline_start_time = time.time()

        # Start monitoring
        self.monitor.start_monitoring()
        self.resource_monitor.start_monitoring()

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

            # Stop monitoring and get metrics
            monitor_metrics = self.monitor.stop_monitoring()
            self.resource_monitor.stop_monitoring()

            # Generate final report
            final_metrics = {
                "stages_completed": len(self.results),
                "stages_failed": len(self.errors),
                "total_stages": len(stages_to_run),
                "execution_time": total_execution_time,
                "monitoring_metrics": monitor_metrics,
                "results": self.results,
                "errors": self.errors,
            }

            self.logger.info(f"[SUCCESS] Pipeline completed successfully!")
            self.logger.info(f"[METRICS] Final metrics: {final_metrics}")

            # Generate and save monitoring report
            if self.config.get("monitoring.enabled", True):
                report = create_monitoring_report(self.monitor, self.resource_monitor)
                self.logger.info(f"[REPORT] Monitoring report generated")

                # Save metrics to file
                metrics_file = self.monitor.save_metrics()
                if metrics_file:
                    self.logger.info(f"[METRICS] Metrics saved to: {metrics_file}")

            return final_metrics

        except Exception as e:
            # Stop monitoring on error
            self.monitor.stop_monitoring()
            self.resource_monitor.stop_monitoring()

            # Calculate total execution time even on error
            pipeline_end_time = time.time()
            total_execution_time = pipeline_end_time - pipeline_start_time

            final_metrics = {
                "stages_completed": len(self.results),
                "stages_failed": len(self.errors),
                "total_stages": len(stages_to_run),
                "execution_time": total_execution_time,
                "results": self.results,
                "errors": self.errors,
                "pipeline_error": str(e),
            }

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
            "total_errors": len(self.errors),
        }


def main():
    """Main entry point for pipeline orchestration."""
    import argparse

    parser = argparse.ArgumentParser(description="Data Pipeline Orchestrator")
    parser.add_argument("--config", default="app/pipeline.yaml", help="Configuration file path")
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
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(f"Stages completed: {result.get('stages_completed', 0)}")
        print(f"Stages failed: {result.get('stages_failed', 0)}")
        print(f"Total execution time: {result.get('execution_time', 0):.2f}s")

        if result.get("errors"):
            print("\nErrors:")
            for error in result["errors"]:
                print(f"  - {error['stage']}: {error['error']}")

        print("=" * 60)

        # Exit with appropriate code
        if result.get("stages_failed", 0) > 0:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
