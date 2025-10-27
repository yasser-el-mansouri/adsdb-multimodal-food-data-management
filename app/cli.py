"""
Command Line Interface for Data Pipeline

This module provides a user-friendly CLI interface for running the data pipeline.
"""

import sys
import subprocess
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from app.utils import PipelineConfig, Logger

app = typer.Typer(help="Data Pipeline CLI")
console = Console()


@app.command()
def run(
    stages: Optional[str] = typer.Option(None, "--stages", "-s", help="Comma-separated list of stages to run"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Single stage to run"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run in dry-run mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    config: str = typer.Option("app/config/pipeline.yaml", "--config", "-c", help="Configuration file path")
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
        if config != "app/config/pipeline.yaml":
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
            border_style="blue"
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
            from app.utils import validate_config
            issues = validate_config(config)
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
        
        # Run unit tests
        console.print("\n[bold]Running unit tests...[/bold]")
        result = subprocess.run([sys.executable, "-m", "pytest", "app/tests/unit/", "-v"], 
                              cwd=Path.cwd())
        
        if result.returncode != 0:
            console.print("[red]✗[/red] Unit tests failed")
            sys.exit(1)
        
        console.print("[green]✓[/green] Unit tests passed")
        
        # Run integration tests
        console.print("\n[bold]Running integration tests...[/bold]")
        result = subprocess.run([sys.executable, "-m", "pytest", "app/tests/integration/", "-v"], 
                              cwd=Path.cwd())
        
        if result.returncode != 0:
            console.print("[red]✗[/red] Integration tests failed")
            sys.exit(1)
        
        console.print("[green]✓[/green] Integration tests passed")
        console.print("\n[bold green]✓ All tests passed![/bold green]")
    
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def monitor():
    """Show real-time pipeline monitoring."""
    try:
        console.print("[bold blue]Pipeline monitoring (press Ctrl+C to stop)...[/bold blue]")
        
        # This would typically connect to a monitoring system
        # For now, just show a placeholder
        console.print("Monitoring functionality would be implemented here")
        console.print("This could include:")
        console.print("- Real-time execution status")
        console.print("- Resource usage metrics")
        console.print("- Error tracking")
        console.print("- Performance metrics")
    
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Monitoring stopped by user[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def init():
    """Initialize the pipeline environment."""
    try:
        console.print("[bold blue]Initializing pipeline environment...[/bold blue]")
        
        # Check if .env file exists
        env_file = Path(".env")
        env_sample = Path("app/.env.sample")
        
        if not env_file.exists() and env_sample.exists():
            console.print(f"[yellow]⚠[/yellow] .env file not found. Please copy {env_sample} to .env and configure it.")
        
        # Check if config directory exists
        config_dir = Path("app/config")
        if not config_dir.exists():
            console.print(f"[red]✗[/red] Config directory not found: {config_dir}")
            sys.exit(1)
        
        console.print("[green]✓[/green] Config directory found")
        
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
