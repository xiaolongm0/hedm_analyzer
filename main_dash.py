#!/usr/bin/env python3
"""
Main entry point for HEDM Analyzer Dash GUI
"""

import logging

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

print("=" * 70)
print("HEDM Analyzer Dash GUI")
print("=" * 70)
print("\nStarting app on http://localhost:8050")
print("\nOpen your browser and navigate to: http://localhost:8050")
print("=" * 70)

from gui_dash.app import app

if __name__ == '__main__':
    logger.info("Starting Dash application server")
    app.run_server(debug=True, host='127.0.0.1', port=8050)
