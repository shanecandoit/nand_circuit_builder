"""
CircuitBuilder - Main Entry Point

This file serves as the main entry point for running experiments and tests.
Individual test suites are organized in the experiments/ directory.

Quick Start:
    python main.py                  # Run all baseline tests
    python main.py --test <name>    # Run specific test
    python main.py --experiment <name>  # Run specific experiment

Examples:
    python experiments/baseline_tests.py      # All baseline tests
    python experiments/circle_area_gateCount.py  # Gate count comparison
"""

import sys
import os


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        if sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print(__doc__)
            print("\nAvailable experiments:")
            print("  - baseline_tests: Run all standard tests")
            print("  - circle_area_gateCount: Compare gate counts for circle area")
            return
        elif sys.argv[1] == '--experiment':
            if len(sys.argv) < 3:
                print("Usage: python main.py --experiment <name>")
                return
            exp_name = sys.argv[2]
            os.system(f"python experiments/{exp_name}.py")
            return
    
    # Default: run baseline tests
    print("Running baseline tests...")
    print("For more options, use: python main.py --help")
    print()
    os.system("python experiments/baseline_tests.py")


if __name__ == '__main__':
    main()
