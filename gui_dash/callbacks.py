"""
Callbacks for HEDM Analyzer Dash GUI application
Handles all user interactions and data updates
"""

import json
import os
from dash import callback, Input, Output, State, ctx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import logging

from gui_dash.utils.image_processing import prepare_image_for_display

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def find_file_path(filename):
    """
    Search for full file path of a given filename.
    Searches common directories and recursively searches home directory.
    """
    if not filename:
        return None

    # If it's already a full path that exists, return it
    if os.path.isfile(filename):
        return filename

    # Search common directories first
    search_dirs = [
        os.path.expanduser('~/Desktop'),
        os.path.expanduser('~/Documents'),
        os.path.expanduser('~/Documents/GitHub'),
        os.path.expanduser('~/Documents/GitHub/hedm_analyzer'),
        os.path.expanduser('~/Documents/GitHub/hedm_analyzer/examples'),
        os.path.expanduser('~/Downloads'),
        os.path.expanduser('~/'),
        os.getcwd(),
    ]

    # Check common directories
    for directory in search_dirs:
        if os.path.isdir(directory):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                logger.info(f"[FILE SEARCH] Found file at: {full_path}")
                return full_path

    # Recursive search in home directory (max 5 levels)
    home_dir = os.path.expanduser('~')
    skip_dirs = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.pytest_cache'}

    def recursive_search(path, name, depth=0, max_depth=5):
        if depth > max_depth:
            return None
        try:
            for entry in os.listdir(path):
                if entry.startswith('.') and entry not in {'.gitignore', '.github'}:
                    continue
                if entry in skip_dirs:
                    continue
                full_entry = os.path.join(path, entry)
                if entry == name and os.path.isfile(full_entry):
                    logger.info(f"[FILE SEARCH] Found file at: {full_entry}")
                    return full_entry
                if os.path.isdir(full_entry):
                    result = recursive_search(full_entry, name, depth + 1, max_depth)
                    if result:
                        return result
        except (PermissionError, OSError):
            pass
        return None

    result = recursive_search(home_dir, filename)
    if result:
        return result

    logger.warning(f"[FILE SEARCH] File not found: {filename}")
    return None


def register_callbacks(app, data_handler, analysis_engine):
    """Register all Dash callbacks"""

    # Callback 0: Handle file browse (search for file path)
    @callback(
        Output('file-path-input', 'value'),
        Input('file-browse-upload', 'filename'),
        prevent_initial_call=True
    )
    def handle_file_browse(filename):
        logger.debug(f"[CALLBACK 0] File browse: {filename}")
        if not filename:
            return ""

        full_path = find_file_path(filename)
        return full_path if full_path else filename

    # Callback 1: Load file from disk
    @callback(
        [Output('data-store', 'data'),
         Output('current-frame-idx', 'data', allow_duplicate=True),
         Output('rois-store', 'data', allow_duplicate=True),
         Output('upload-status', 'children')],
        Input('btn-load-file', 'n_clicks'),
        State('file-path-input', 'value'),
        prevent_initial_call=True
    )
    def load_file_from_disk(n_clicks, file_path):
        logger.debug(f"[CALLBACK 1] Load file: {file_path}")

        if not file_path:
            return None, 0, [], "Error: No file path provided"

        if not os.path.isfile(file_path):
            return None, 0, [], f"Error: File not found: {file_path}"

        try:
            data_handler.load_hdf5(file_path)
            analysis_engine.clear_rois()

            metadata = {
                'filename': os.path.basename(file_path),
                'filepath': file_path,
                'num_frames': data_handler.data.shape[0] if data_handler.data is not None else 0,
                'frame_shape': data_handler.data.shape[1:] if data_handler.data is not None else (0, 0),
            }

            status = f"Loaded: {metadata['filename']} ({metadata['num_frames']} frames, {metadata['frame_shape']})"
            logger.info(f"[CALLBACK 1] {status}")

            return metadata, 0, [], status
        except Exception as e:
            error_msg = f"Error loading file: {str(e)}"
            logger.error(f"[CALLBACK 1] {error_msg}")
            return None, 0, [], error_msg

    # Callback 2: Update image display
    @callback(
        Output('image-display', 'figure'),
        [Input('current-frame-idx', 'data'),
         Input('sat-threshold-input', 'value'),
         Input('highlight-saturation-check', 'value'),
         Input('lower-threshold-input', 'value'),
         Input('rois-store', 'data')],
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_image_display(frame_idx, sat_threshold, highlight_sat, lower_threshold, rois, metadata):
        logger.debug(f"[CALLBACK 2] Update image display: frame {frame_idx}")

        if not metadata or data_handler.data is None or frame_idx >= data_handler.data.shape[0]:
            return go.Figure()

        try:
            frame = data_handler.get_frame(frame_idx)
            should_highlight = 1 in (highlight_sat or [])

            # Prepare image with saturation highlighting
            rgb_image = prepare_image_for_display(
                frame,
                saturation_threshold=sat_threshold or 65535,
                saturation_highlight=should_highlight,
                lower_threshold=lower_threshold or 0
            )

            # Create figure
            fig = px.imshow(rgb_image, color_continuous_scale='gray')

            # Add ROI shapes from rois store
            if rois:
                for roi in rois:
                    fig.add_shape(
                        type="rect",
                        x0=roi['x0'], y0=roi['y0'],
                        x1=roi['x1'], y1=roi['y1'],
                        line=dict(color="red", width=2),
                        fill=None
                    )

            # Configure for ROI drawing
            fig.update_layout(
                dragmode='drawrect',
                newshape={'line': {'color': 'red', 'width': 2}},
                height=700,
                showlegend=False,
                hovermode=False
            )

            logger.debug(f"[CALLBACK 2] Image displayed successfully")
            return fig
        except Exception as e:
            logger.error(f"[CALLBACK 2] Error: {str(e)}")
            return go.Figure()

    # Callback 3: Update frame statistics
    @callback(
        [Output('frame-label', 'children'),
         Output('range-label', 'children'),
         Output('saturation-label', 'children')],
        [Input('current-frame-idx', 'data'),
         Input('sat-threshold-input', 'value'),
         Input('lower-threshold-input', 'value')],
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def update_frame_stats(frame_idx, sat_threshold, lower_threshold, metadata):
        logger.debug(f"[CALLBACK 3] Update frame stats: frame {frame_idx}")

        if not metadata or data_handler.data is None or frame_idx >= data_handler.data.shape[0]:
            return "", "", ""

        try:
            frame = data_handler.get_frame(frame_idx)
            skip_frames = 0  # TODO: Get from state if needed

            frame_label = f"Frame {frame_idx + skip_frames}/{data_handler.data.shape[0] - 1 + skip_frames}"
            range_label = f"Range: {frame.min()} - {frame.max()}"

            sat_threshold = sat_threshold or 65535
            sat_pixels = np.sum(frame >= sat_threshold)
            sat_percent = 100 * sat_pixels / frame.size if frame.size > 0 else 0
            sat_label = f"Saturated: {sat_pixels} pixels ({sat_percent:.2f}%)"

            return frame_label, range_label, sat_label
        except Exception as e:
            logger.error(f"[CALLBACK 3] Error: {str(e)}")
            return "", "", ""

    # Callback 4: Navigate frames (combined prev/next)
    @callback(
        Output('current-frame-idx', 'data', allow_duplicate=True),
        [Input('btn-prev-frame', 'n_clicks'),
         Input('btn-next-frame', 'n_clicks')],
        State('current-frame-idx', 'data'),
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def navigate_frames(prev_clicks, next_clicks, current_idx, metadata):
        logger.debug(f"[CALLBACK 4] Navigate frames: {ctx.triggered_id}")

        if not metadata or data_handler.data is None:
            return 0

        new_idx = current_idx or 0

        if ctx.triggered_id == 'btn-prev-frame':
            new_idx = max(0, new_idx - 1)
        elif ctx.triggered_id == 'btn-next-frame':
            new_idx = min(data_handler.data.shape[0] - 1, new_idx + 1)

        logger.debug(f"[CALLBACK 4] Frame index: {new_idx}")
        return new_idx

    # Callback 5: Update ROI list display
    @callback(
        Output('roi-list-display', 'children'),
        Input('rois-store', 'data'),
        prevent_initial_call=True
    )
    def update_roi_list(rois):
        logger.debug(f"[CALLBACK 5] Update ROI list: {len(rois) if rois else 0} ROIs")

        if not rois:
            return html.Small("No ROIs defined yet", className="text-muted")

        roi_items = []
        for roi in rois:
            roi_text = f"{roi['name']}: ({roi['x0']:.0f}, {roi['y0']:.0f}) - ({roi['x1']:.0f}, {roi['y1']:.0f})"
            roi_items.append(html.Div(roi_text, className="small text-break"))

        return roi_items

    # Callback 6: Manage ROIs (combined clear/capture)
    @callback(
        Output('rois-store', 'data', allow_duplicate=True),
        [Input('btn-clear-rois', 'n_clicks'),
         Input('image-display', 'relayoutData')],
        State('rois-store', 'data'),
        State('roi-name-input', 'value'),
        prevent_initial_call=True
    )
    def manage_rois(clear_clicks, relayout_data, rois, roi_name):
        logger.debug(f"[CALLBACK 6] Manage ROIs: {ctx.triggered_id}")

        rois = rois or []

        if ctx.triggered_id == 'btn-clear-rois':
            logger.info("[CALLBACK 6] Clearing all ROIs")
            analysis_engine.clear_rois()
            return []

        elif ctx.triggered_id == 'image-display':
            # Check if new shape was drawn
            if relayout_data and 'shapes' in relayout_data and relayout_data['shapes']:
                shapes = relayout_data['shapes']
                if shapes and len(shapes) > len(rois):
                    # New shape added
                    new_shape = shapes[-1]
                    roi_name = roi_name or f"ROI_{len(rois) + 1}"

                    new_roi = {
                        'name': roi_name,
                        'x0': new_shape.get('x0', 0),
                        'y0': new_shape.get('y0', 0),
                        'x1': new_shape.get('x1', 0),
                        'y1': new_shape.get('y1', 0),
                    }
                    rois.append(new_roi)
                    logger.info(f"[CALLBACK 6] New ROI added: {roi_name}")

        return rois

    # Callback 7: Analyze frame
    @callback(
        [Output('results-text', 'value'),
         Output('histogram-plot', 'figure')],
        Input('btn-analyze-frame', 'n_clicks'),
        [State('current-frame-idx', 'data'),
         State('sat-threshold-input', 'value'),
         State('lower-threshold-input', 'value'),
         State('rois-store', 'data')],
        State('data-store', 'data'),
        prevent_initial_call=True
    )
    def analyze_frame(n_clicks, frame_idx, sat_threshold, lower_threshold, rois, metadata):
        logger.debug(f"[CALLBACK 7] Analyze frame {frame_idx}")

        if not metadata or data_handler.data is None or frame_idx >= data_handler.data.shape[0]:
            return "Error: No data loaded", go.Figure()

        try:
            frame = data_handler.get_frame(frame_idx)
            sat_threshold = sat_threshold or 65535
            lower_threshold = lower_threshold or 0

            # Filter frame based on thresholds
            filtered_frame = frame.copy().astype(np.float64)
            if lower_threshold > 0:
                filtered_frame[filtered_frame < lower_threshold] = 0

            # Calculate statistics
            results = {
                'frame_index': int(frame_idx),
                'overall': {
                    'min': float(filtered_frame.min()),
                    'max': float(filtered_frame.max()),
                    'mean': float(np.mean(filtered_frame[filtered_frame > 0])) if np.any(filtered_frame > 0) else 0,
                    'median': float(np.median(filtered_frame[filtered_frame > 0])) if np.any(filtered_frame > 0) else 0,
                    'std': float(np.std(filtered_frame[filtered_frame > 0])) if np.any(filtered_frame > 0) else 0,
                    'saturation_count': int(np.sum(frame >= sat_threshold)),
                    'saturation_percent': float(100 * np.sum(frame >= sat_threshold) / frame.size),
                },
                'rois': []
            }

            # Calculate ROI statistics
            rois = rois or []
            for roi in rois:
                y0, y1 = int(roi['y0']), int(roi['y1'])
                x0, x1 = int(roi['x0']), int(roi['x1'])
                y0, y1 = min(y0, y1), max(y0, y1)
                x0, x1 = min(x0, x1), max(x0, x1)

                roi_data = filtered_frame[y0:y1, x0:x1]
                if roi_data.size > 0:
                    valid_data = roi_data[roi_data > 0]
                    results['rois'].append({
                        'name': roi['name'],
                        'min': float(np.min(roi_data)),
                        'max': float(np.max(roi_data)),
                        'mean': float(np.mean(valid_data)) if len(valid_data) > 0 else 0,
                        'median': float(np.median(valid_data)) if len(valid_data) > 0 else 0,
                        'std': float(np.std(valid_data)) if len(valid_data) > 0 else 0,
                        'saturation_count': int(np.sum(roi_data >= sat_threshold)),
                    })

            results_text = json.dumps(results, indent=2)

            # Create histogram figure
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Overall Frame', 'ROI 1', 'ROI 2', 'ROI 3']
            )

            # Overall histogram
            overall_data = filtered_frame[filtered_frame > 0].flatten()
            if len(overall_data) > 0:
                fig.add_trace(
                    go.Histogram(
                        x=overall_data,
                        name='Overall',
                        marker_color='blue',
                        nbinsx=100
                    ),
                    row=1, col=1
                )

            # ROI histograms
            roi_positions = [(1, 2), (2, 1), (2, 2)]
            for idx, (roi, pos) in enumerate(zip(rois[:3], roi_positions)):
                y0, y1 = int(roi['y0']), int(roi['y1'])
                x0, x1 = int(roi['x0']), int(roi['x1'])
                y0, y1 = min(y0, y1), max(y0, y1)
                x0, x1 = min(x0, x1), max(x0, x1)

                roi_data = filtered_frame[y0:y1, x0:x1]
                valid_roi = roi_data[roi_data > 0].flatten()

                if len(valid_roi) > 0:
                    fig.add_trace(
                        go.Histogram(
                            x=valid_roi,
                            name=roi['name'],
                            marker_color='green',
                            nbinsx=50
                        ),
                        row=pos[0], col=pos[1]
                    )

            fig.update_layout(height=700, showlegend=False, hovermode='closest')

            logger.info("[CALLBACK 7] Analysis complete")
            return results_text, fig
        except Exception as e:
            logger.error(f"[CALLBACK 7] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}", go.Figure()


# Import html for use in update_roi_list
from dash import html
