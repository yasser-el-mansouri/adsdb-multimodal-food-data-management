#!/usr/bin/env python3
"""
Setup script for the data pipeline operations environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        "boto3", "chromadb", "huggingface_hub", "typer", "rich", "pytest"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r app/requirements.txt")
        return False
    
    return True


def setup_environment():
    """Set up the environment."""
    print("🚀 Setting up data pipeline operations environment...")
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check if we're in the right directory
    if not Path("app").exists():
        print("❌ Please run this script from the project root directory")
        return False
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Create necessary directories
    directories = [
        "app/config",
        "app/utils", 
        "app/zones",
        "app/tests/unit",
        "app/tests/integration",
        "app/monitoring"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory {directory} ready")
    
    # Check if .env file exists
    if not Path(".env").exists():
        if Path("app/.env.sample").exists():
            print("⚠️  .env file not found. Please copy app/.env.sample to .env and configure it")
        else:
            print("❌ app/.env.sample not found")
            return False
    else:
        print("✅ .env file found")
    
    # Check if config file exists
    if not Path("app/config/pipeline.yaml").exists():
        print("❌ app/config/pipeline.yaml not found")
        return False
    else:
        print("✅ Configuration file found")
    
    print("\n🎉 Environment setup completed successfully!")
    print("\nNext steps:")
    print("1. Configure your .env file with actual values")
    print("2. Run: python -m app.cli validate")
    print("3. Run: python -m app.cli run --dry-run")
    print("4. Run: python -m app.cli run")
    
    return True


def main():
    """Main setup function."""
    try:
        success = setup_environment()
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
