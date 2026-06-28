import sys
import os

# The app requires SESSION_SECRET_KEY (it signs the login cookie; no fallback). Tests do no
# real OAuth, so a dummy value is fine — set it before any test imports backend.main.
os.environ.setdefault('SESSION_SECRET_KEY', 'test-only-session-secret')

# Add the testing_files/ directory to sys.path so that benchmark_helpers
# can be imported by the benchmark_* test modules without a package prefix.
# Pytest's working directory is the project root, so this directory would
# not otherwise be on the path.
sys.path.insert(0, os.path.dirname(__file__))
