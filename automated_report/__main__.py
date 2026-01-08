"""
Entry point for running automated_report as a module.

Usage:
    python -m automated_report --help
    python -m automated_report --input_dir ./czi_files/ --genotypes WT KO --nuclear_channel 0
"""

from .cli import main

if __name__ == "__main__":
    main()
