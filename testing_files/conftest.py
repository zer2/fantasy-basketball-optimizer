import sys
import os

# Add the testing_files/ directory to sys.path so that benchmark_helpers
# can be imported by the benchmark_* test modules without a package prefix.
# Pytest's working directory is the project root, so this directory would
# not otherwise be on the path.
sys.path.insert(0, os.path.dirname(__file__))
