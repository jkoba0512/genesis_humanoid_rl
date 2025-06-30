#!/usr/bin/env python3
"""
Environment setup and verification tool.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python 3.10 is being used."""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print(f"❌ Python 3.10 required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor} detected")
    return True


def check_uv_installation():
    """Check if uv is installed."""
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ uv detected: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("❌ uv not found. Install with:")
    print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
    return False


def install_dependencies():
    """Install project dependencies with uv."""
    print("📦 Installing dependencies...")
    try:
        result = subprocess.run(
            ["uv", "sync"], check=True, capture_output=True, text=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print(f"Error output: {e.stderr}")
        return False


def verify_imports():
    """Verify key imports work correctly."""
    print("🔍 Verifying imports...")

    imports_to_test = [
        ("genesis", "Genesis physics engine"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
    ]

    all_good = True
    for module, name in imports_to_test:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_good = False

    return all_good


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")

    directories = [
        "logs",
        "models",
        "data",
        "experiments",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created {directory}/")


def setup_git_hooks():
    """Setup git hooks for code quality."""
    hooks_dir = ".git/hooks"
    if not os.path.exists(hooks_dir):
        print("⚠️  Git repository not initialized")
        return

    pre_commit_hook = f"""#!/bin/bash
# Auto-format code before commit
echo "Running code formatting..."
uv run black src/ scripts/ tools/
uv run isort src/ scripts/ tools/
echo "Code formatting complete"
"""

    hook_path = os.path.join(hooks_dir, "pre-commit")
    with open(hook_path, "w") as f:
        f.write(pre_commit_hook)

    os.chmod(hook_path, 0o755)
    print("✅ Git pre-commit hook installed")


def main():
    """Main setup function."""
    print("🚀 Genesis Humanoid RL Environment Setup")
    print("=" * 50)

    checks = [
        ("Python Version", check_python_version),
        ("uv Installation", check_uv_installation),
    ]

    # Run initial checks
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        if not check_func():
            print(f"\n❌ Setup failed at: {name}")
            print("Please fix the issue and run setup again.")
            sys.exit(1)

    # Install and verify
    setup_steps = [
        ("Installing Dependencies", install_dependencies),
        ("Verifying Imports", verify_imports),
        ("Creating Directories", create_directories),
        ("Setting up Git Hooks", setup_git_hooks),
    ]

    for name, step_func in setup_steps:
        print(f"\n{name}...")
        if not step_func():
            print(f"⚠️  {name} had issues but continuing...")

    print("\n" + "=" * 50)
    print("🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("1. Test installation: uv run python tools/verify_installation.py")
    print(
        "2. Quick training test: uv run python scripts/train_sb3.py --config configs/test.json"
    )
    print("3. Monitor training: tensorboard --logdir ./logs")
    print("4. Read documentation: docs/README.md")
    print("\n🤖 Ready to train humanoid robots!")


if __name__ == "__main__":
    main()
