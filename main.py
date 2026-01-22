"""
CircuitBuilder - Main Entry Point

This file serves as the main entry point for running experiments and tests.
Individual test suites are organized in the experiments/ directory.

Quick Start:
    python main.py                  # Run all experiments
    python main.py --experiment <name>  # Run specific experiment

Examples:
    python experiments/baseline_tests.py      # All baseline tests
    python experiments/circle_area_gateCount.py  # Gate count comparison
    python experiments/real_diabetes.py       # Diabetes regression experiment
    python experiments/real_housing.py        # Housing prices regression experiment
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
            print("  - real_diabetes: Diabetes regression experiment (R² > 0.4)")
            print("  - real_housing: Housing prices regression experiment (R² > 0.6)")
            print("\nNote: Running 'python main.py' without arguments will run ALL experiments.")
            return
        elif sys.argv[1] == '--experiment':
            if len(sys.argv) < 3:
                print("Usage: python main.py --experiment <name>")
                return
            exp_name = sys.argv[2]
            os.system(f"python experiments/{exp_name}.py")
            return
    
    # Default: run all experiments
    experiments = [
        'baseline_tests',
        'circle_area_gateCount',
        'real_diabetes',
        'real_housing'
    ]
    
    print("="*70)
    print("Running All Experiments")
    print("="*70)
    print(f"\nFound {len(experiments)} experiments to run:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    print("\n" + "="*70)
    
    results = {}
    for exp_name in experiments:
        print(f"\n{'='*70}")
        print(f"Running: {exp_name}")
        print('='*70)
        exit_code = os.system(f"python experiments/{exp_name}.py")
        results[exp_name] = exit_code == 0
        print(f"\n{'='*70}")
        print(f"Completed: {exp_name} {'[OK]' if exit_code == 0 else '[ERROR]'}")
        print('='*70)
    
    # Summary
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print('='*70)
    for exp_name, success in results.items():
        status = "[OK]" if success else "[ERROR]"
        print(f"  {exp_name:30s}: {status}")
    print('='*70)


if __name__ == '__main__':
    main()
