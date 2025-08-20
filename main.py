#!/usr/bin/env python3
"""
HEDM X-ray Image Analyzer - Main Entry Point
"""

import sys
import os
import logging

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from gui.main_window import main
    
    if __name__ == "__main__":
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('hedm_analyzer.log'),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("Starting HEDM X-ray Image Analyzer")
        
        try:
            main()
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            sys.exit(1)

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required packages are installed:")
    print("pip install numpy pillow matplotlib h5py scikit-image")
    sys.exit(1)