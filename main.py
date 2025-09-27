#!/usr/bin/env python3
"""
Main entry point for the Archie AI Agent application.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.chat import main

if __name__ == "__main__":
    main()
