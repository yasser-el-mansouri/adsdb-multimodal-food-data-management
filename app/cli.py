"""
Command Line Interface for Data Pipeline

This module provides a user-friendly CLI interface for running the data pipeline.
"""

import os
import subprocess

# Import configuration and utilities directly to avoid dependency issues
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
import config

PipelineConfig = config.PipelineConfig

app = typer.Typer(help="Data Pipeline CLI")
console = Console()


@app.command()
def run(
    stages: Optional[str] = typer.Option(
        None, "--stages", "-s", help="Comma-separated list of stages to run"
    ),
    stage: Optional[str] = typer.Option(None, "--stage", help="Single stage to run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run in dry-run mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config: str = typer.Option(
        "app/pipeline.yaml", "--config", "-c", help="Configuration file path"
    ),
):
    """Run the data pipeline."""
    try:
        # Build command
        cmd = [sys.executable, "app/orchestrate.py"]

        if stages:
            cmd.extend(["--stages"] + stages.split(","))
        if stage:
            cmd.extend(["--stage", stage])
        if dry_run:
            cmd.append("--dry-run")
        if verbose:
            cmd.append("--verbose")
        if config != "app/pipeline.yaml":
            cmd.extend(["--config", config])

        console.print(f"[bold blue]Running command:[/bold blue] {' '.join(cmd)}")

        # Run the orchestrator
        result = subprocess.run(cmd, cwd=Path.cwd())
        sys.exit(result.returncode)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def status():
    """Show pipeline status and available stages."""
    try:
        config = PipelineConfig()

        # Create table for stages
        table = Table(title="Pipeline Stages")
        table.add_column("Stage", style="cyan")
        table.add_column("Description", style="magenta")
        table.add_column("Status", style="green")

        stages = [
            ("temporal_landing", "Ingest raw data from external sources"),
            ("persistent_landing", "Organize data by type and apply naming conventions"),
            ("formatted_documents", "Join recipes and remove irrelevant data"),
            ("formatted_images", "Process and organize images"),
            ("trusted_images", "Extract recipe IDs and copy filtered images"),
            ("trusted_documents", "Filter documents and apply quality controls"),
            ("exploitation_documents", "Generate embeddings and store in ChromaDB"),
            ("exploitation_images", "Generate image embeddings and store in ChromaDB"),
            ("task1_retrieval", "Task 1: Multimodal retrieval operations"),
        ]

        for stage_name, description in stages:
            table.add_row(stage_name, description, "Available")

        console.print(table)

        # Show configuration info
        config_info = Panel(
            f"Configuration file: {config.config_path}\n"
            f"Dry run mode: {config.get('pipeline.dry_run', False)}\n"
            f"Overwrite mode: {config.get('pipeline.overwrite', True)}\n"
            f"Batch size: {config.get('pipeline.batch_size', 256)}\n"
            f"Log level: {config.get('monitoring.log_level', 'INFO')}",
            title="Configuration",
            border_style="blue",
        )
        console.print(config_info)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def validate():
    """Validate configuration and environment."""
    try:
        config = PipelineConfig()

        console.print("[bold blue]Validating configuration and environment...[/bold blue]")

        # Check configuration file
        if Path(config.config_path).exists():
            console.print(f"[green]✓[/green] Configuration file found: {config.config_path}")
        else:
            console.print(f"[red]✗[/red] Configuration file not found: {config.config_path}")

        # Check environment variables
        required_env_vars = ["MINIO_USER", "MINIO_PASSWORD", "MINIO_ENDPOINT"]
        env_status = True

        for var in required_env_vars:
            if config.get_env(var):
                console.print(f"[green]✓[/green] Environment variable {var} is set")
            else:
                console.print(f"[red]✗[/red] Environment variable {var} is missing")
                env_status = False

        # Check optional environment variables
        optional_env_vars = ["HF_TOKEN", "HF_ORGA", "HF_DATASET", "CHROMA_PERSIST_DIR"]
        for var in optional_env_vars:
            if config.get_env(var):
                console.print(f"[green]✓[/green] Optional environment variable {var} is set")
            else:
                console.print(f"[yellow]⚠[/yellow] Optional environment variable {var} is not set")

        # Validate configuration
        issues = []
        try:
            issues = config.validate_config(PipelineConfig())
        except Exception as e:
            console.print(f"[red]✗[/red] Configuration validation failed: {e}")
            return

        if issues:
            console.print("[red]✗[/red] Configuration validation issues found:")
            for issue in issues:
                console.print(f"  - {issue}")
        else:
            console.print("[green]✓[/green] Configuration validation passed")

        # Overall status
        if env_status and not issues:
            console.print("\n[bold green]✓ All validations passed![/bold green]")
        else:
            console.print("\n[bold red]✗ Some validations failed![/bold red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def test():
    """Run tests for the pipeline."""
    try:
        console.print("[bold blue]Running pipeline tests...[/bold blue]")

        # Check if pytest is available
        try:
            import pytest
        except ImportError:
            console.print(
                "[red]✗[/red] pytest is not installed. Please install it with: pip install pytest"
            )
            sys.exit(1)

        # Run unit tests
        console.print("\n[bold]Running unit tests...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "app/tests/unit/", "-v"], cwd=Path.cwd()
        )

        if result.returncode != 0:
            console.print("[red]✗[/red] Unit tests failed")
            sys.exit(1)

        console.print("[green]✓[/green] Unit tests passed")

        # Run integration tests
        console.print("\n[bold]Running integration tests...[/bold]")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "app/tests/integration/", "-v"], cwd=Path.cwd()
        )

        if result.returncode != 0:
            console.print("[red]✗[/red] Integration tests failed")
            sys.exit(1)

        console.print("[green]✓[/green] Integration tests passed")
        console.print("\n[bold green]✓ All tests passed![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def metrics():
    """Show current system metrics."""
    try:
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
        from monitoring import get_system_info, utc_timestamp

        console.print("[bold blue]📊 Current System Metrics[/bold blue]")
        console.print("=" * 50)

        info = get_system_info()
        timestamp = utc_timestamp()

        console.print(f"Timestamp: {timestamp}")
        console.print(f"CPU Count: {info['cpu_count']}")
        console.print(f"CPU Usage: {info['cpu_percent']:.1f}%")
        console.print(f"Memory Usage: {info['memory_percent']:.1f}%")
        console.print(f"Memory Used: {info['memory_used'] / (1024**3):.2f} GB")
        console.print(f"Memory Available: {info['memory_available'] / (1024**3):.2f} GB")
        console.print(f"Disk Usage: {info['disk_percent']:.1f}%")
        console.print(f"Disk Used: {info['disk_used'] / (1024**3):.2f} GB")
        console.print(f"Disk Free: {info['disk_free'] / (1024**3):.2f} GB")

        if not info["psutil_available"]:
            console.print(
                "\n[yellow]⚠️  psutil not available - install with: pip install psutil[/yellow]"
            )

    except Exception as e:
        console.print(f"[bold red]❌ Error getting metrics:[/bold red] {e}")


@app.command()
def report():
    """Generate monitoring report from latest metrics file."""
    try:
        import glob
        import json
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))
        from monitoring import create_monitoring_report

        # Find latest metrics file
        metrics_files = glob.glob("pipeline_metrics_*.json")
        if not metrics_files:
            console.print(
                "[red]❌ No metrics files found. Run the pipeline first to generate metrics.[/red]"
            )
            return

        latest_file = max(metrics_files, key=os.path.getctime)

        with open(latest_file, "r") as f:
            data = json.load(f)

        console.print(f"[bold blue]📊 Monitoring Report from {latest_file}[/bold blue]")
        console.print("=" * 60)

        # Pipeline summary
        pipeline_metrics = data.get("pipeline_metrics", {})
        console.print(f"Execution Time: {pipeline_metrics.get('execution_time', 0):.2f} seconds")
        console.print(f"Memory Usage: {pipeline_metrics.get('memory_usage', 0) / (1024**2):.1f} MB")
        console.print(f"Disk Usage: {pipeline_metrics.get('disk_usage', 0) / (1024**2):.1f} MB")
        console.print(f"Peak Memory: {pipeline_metrics.get('peak_memory', 0) / (1024**3):.2f} GB")
        console.print(f"Current Memory: {pipeline_metrics.get('current_memory_percent', 0):.1f}%")
        console.print(f"Current Disk: {pipeline_metrics.get('current_disk_percent', 0):.1f}%")
        console.print(f"CPU Usage: {pipeline_metrics.get('cpu_percent', 0):.1f}%")

        # Stage details
        stage_metrics = data.get("stage_metrics", {})
        if stage_metrics:
            console.print(f"\n[bold green]📈 Stage Performance:[/bold green]")
            for stage_name, stage_data in stage_metrics.items():
                duration = stage_data.get("duration", 0)
                memory = stage_data.get("memory_usage", 0)
                success = stage_data.get("success", True)
                status = "✅" if success else "❌"
                console.print(
                    f"  {status} {stage_name}: {duration:.2f}s, {memory / (1024**2):.1f} MB"
                )

        # Errors
        errors = data.get("pipeline_metrics", {}).get("errors", [])
        if errors:
            console.print(f"\n[bold red]❌ Errors ({len(errors)}):[/bold red]")
            for error in errors:
                console.print(
                    f"  - {error.get('stage', 'unknown')}: {error.get('error', 'unknown error')}"
                )

    except Exception as e:
        console.print(f"[bold red]❌ Error generating report:[/bold red] {e}")


@app.command()
def init():
    """Initialize the pipeline environment."""
    try:
        console.print("[bold blue]Initializing pipeline environment...[/bold blue]")

        # Check if .env file exists
        env_file = Path(".env")
        env_sample = Path("app/.env.sample")

        if not env_file.exists() and env_sample.exists():
            console.print(
                f"[yellow]⚠[/yellow] .env file not found. Please copy {env_sample} to .env and configure it."
            )

        # Check if config file exists
        config_file = Path("app/pipeline.yaml")
        if not config_file.exists():
            console.print(f"[red]✗[/red] Config file not found: {config_file}")
            sys.exit(1)

        console.print("[green]✓[/green] Config file found")

        # Check if required directories exist
        required_dirs = ["app/zones", "app/utils", "app/tests"]
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                console.print(f"[green]✓[/green] Directory found: {dir_path}")
            else:
                console.print(f"[red]✗[/red] Directory not found: {dir_path}")

        console.print("\n[bold green]✓ Environment initialization completed![/bold green]")
        console.print("\nNext steps:")
        console.print("1. Copy app/.env.sample to .env and configure it")
        console.print("2. Run 'python -m app.cli validate' to check your setup")
        console.print("3. Run 'python -m app.cli run' to start the pipeline")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
