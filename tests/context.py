"""For easy importing of the main library in tests."""

import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import adafilt  # noqa
