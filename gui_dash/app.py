"""
Main Dash application setup for HEDM Analyzer
"""

import dash
import dash_bootstrap_components as dbc
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_handler import DataHandler
from core.analysis_engine import AnalysisEngine
from gui_dash.layouts import create_layout
from gui_dash.callbacks import register_callbacks

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Create module-level instances for single-user deployment
data_handler = DataHandler()
analysis_engine = AnalysisEngine()

# Set the layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app, data_handler, analysis_engine)

# Configure for better performance
app.config.suppress_callback_exceptions = True

# Configure Flask to accept large file uploads (for HDF5 files)
# Set max content length to 1GB (1024*1024*1024 bytes)
app.server.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024

if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1', port=8050)
