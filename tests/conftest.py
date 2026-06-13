import sys
import os

# Ensure the project root is on sys.path so 'import rate_limiter' works
# regardless of the directory from which pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
