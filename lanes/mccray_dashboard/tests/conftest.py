import os
import sys

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "dashboard")
sys.path.insert(0, DASHBOARD_DIR)
