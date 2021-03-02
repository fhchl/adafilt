"""For easy importing of the main library in tests."""
import sys
import os
sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import adafilt  # noqa
