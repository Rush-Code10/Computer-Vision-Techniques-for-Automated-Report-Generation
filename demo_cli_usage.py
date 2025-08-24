#!/usr/bin/env python3
"""
Demo script showing CLI usage examples.
This script demonstrates various ways to use the new CLI system.
"""

import os
import subprocess
import sys


def run_cli_command(cmd_args, description):
    """Run a CLI command and show the results."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: python cli.py {' '.join(cmd_args)}")
    print(f"{'='*60}")
    
    try:
        # Run the command
        result = subprocess.run(
            [sys.executable, "cli.py"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Command timed out (30s limit)")
    except Exception as e:
        print(f"Error running command: {e}")


def main():
    """Run CLI usage demonstrations."""
    print("CLI USAGE DEMONSTRATIONS")
    print("This script shows various ways to use the new CLI system.")
    
    # Check if we have the required image files
    if not (os.path.exists("orlando2010.png") and os.path.exists("orlando2023.png")):
        print("\nNote: Using placeholder image paths for demonstration.")
        print("In real usage, provide actual image file paths.")
        img1, img2 = "image1.jpg", "image2.jpg"
    else:
        img1, img2 = "orlando2010.png", "orlando2023.png"
    
    # Demo 1: Show help
    run_cli_command(["--help"], "Show CLI help")
    
    # Demo 2: List implementations
    run_cli_command(["--list-implementations"], "List available implementations")
    
    # Demo 3: Show current configuration
    run_cli_command(["--show-config"], "Show current configuration")
    
    # Demo 4: Validate configuration
    run_cli_command(["--validate-config"], "Validate configuration")
    
    # Demo 5: Show what would happen with actual images (dry run style)
    if os.path.exists(img1) and os.path.exists(img2):
        print(f"\n{'='*60}")
        print("DEMO: Actual processing would work with these commands:")
        print(f"{'='*60}")
        
        example_commands = [
            f"python cli.py {img1} {img2}",
            f"python cli.py {img1} {img2} -i basic --save",
            f"python cli.py {img1} {img2} --generate-reports --verbose",
            f"python cli.py {img1} {img2} -i advanced -i deep_learning --output-dir my_results",
            f"python cli.py {img1} {img2} --min-area 200 --confidence-threshold 0.7",
        ]
        
        for cmd in example_commands:
            print(f"  • {cmd}")
    
    print(f"\n{'='*60}")
    print("CLI DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    print("The CLI system provides:")
    print("  • Configuration file management")
    print("  • Flexible command-line options")
    print("  • Comprehensive logging and progress tracking")
    print("  • Validation and error handling")
    print("  • Integration with all existing functionality")


if __name__ == "__main__":
    main()