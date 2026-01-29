"""
UI Layout definitions for HEDM Analyzer Dash GUI
"""

import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc


def create_layout():
    """Create the main application layout"""

    return dbc.Container(
        fluid=True,
        children=[
            # Title
            dbc.Row(
                dbc.Col(
                    html.H1("HEDM X-ray Image Analyzer (Web GUI)", className="mb-4 mt-4"),
                    width=12
                )
            ),

            # Main content row with two panels
            dbc.Row(
                [
                    # Left Panel: Controls (25% width)
                    dbc.Col(
                        [
                            # File Input Section
                            dbc.Card(
                                [
                                    dbc.CardHeader("Data Input"),
                                    dbc.CardBody(
                                        [
                                            dbc.Label("HDF5 File Path:", className="fw-bold"),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Input(
                                                        id='file-path-input',
                                                        type='text',
                                                        placeholder='Enter file path or click Browse...',
                                                        className="mb-2"
                                                    ),
                                                ], width=9),
                                                dbc.Col([
                                                    dcc.Upload(
                                                        id='file-browse-upload',
                                                        children=dbc.Button(
                                                            "Browse",
                                                            color='secondary',
                                                            outline=True,
                                                            className="w-100"
                                                        ),
                                                        multiple=False,
                                                        accept='.h5,.hdf5',
                                                    ),
                                                ], width=3),
                                            ]),
                                            dbc.Button(
                                                "Load File from Disk",
                                                id='btn-load-file',
                                                color='primary',
                                                className="w-100 mb-2"
                                            ),
                                            html.Div(id='upload-status', className='text-muted small mt-2'),
                                        ]
                                    ),
                                ],
                                className="mb-3"
                            ),

                            # Parameters Section
                            dbc.Card(
                                [
                                    dbc.CardHeader("Analysis Parameters"),
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Saturation Threshold:", className="fw-bold"),
                                                        dbc.Input(
                                                            id='sat-threshold-input',
                                                            type='number',
                                                            value=65535,
                                                            min=0,
                                                            step=1,
                                                            className="mb-2"
                                                        ),
                                                    ]
                                                )
                                            ),
                                            dbc.Checklist(
                                                id='highlight-saturation-check',
                                                options=[
                                                    {'label': ' Highlight Saturated Pixels (Red)',
                                                     'value': 1}
                                                ],
                                                value=[1],
                                                switch=True,
                                                className="mb-2"
                                            ),
                                            dbc.Row(
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Skip Frames:", className="fw-bold"),
                                                        dbc.Input(
                                                            id='skip-frames-input',
                                                            type='number',
                                                            value=0,
                                                            min=0,
                                                            max=10,
                                                            step=1,
                                                            className="mb-2"
                                                        ),
                                                        html.Small(
                                                            "(Start from frame N+1)",
                                                            className="text-muted"
                                                        ),
                                                    ]
                                                )
                                            ),
                                            dbc.Row(
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Lower Threshold:", className="fw-bold"),
                                                        dbc.Input(
                                                            id='lower-threshold-input',
                                                            type='number',
                                                            value=0,
                                                            min=0,
                                                            step=1,
                                                            className="mb-2"
                                                        ),
                                                        html.Small(
                                                            "(Exclude pixels below this value)",
                                                            className="text-muted"
                                                        ),
                                                    ]
                                                )
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3"
                            ),

                            # ROI Management Section
                            dbc.Card(
                                [
                                    dbc.CardHeader("ROI Management"),
                                    dbc.CardBody(
                                        [
                                            dbc.Label("ROI Name:", className="fw-bold"),
                                            dbc.Input(
                                                id='roi-name-input',
                                                type='text',
                                                placeholder='ROI_1',
                                                className="mb-2"
                                            ),
                                            dbc.Button(
                                                "Clear All ROIs",
                                                id='btn-clear-rois',
                                                color='danger',
                                                outline=True,
                                                size='sm',
                                                className="w-100 mb-2"
                                            ),
                                            html.Div(
                                                id='roi-list-display',
                                                className="border rounded p-2",
                                                style={'maxHeight': '150px', 'overflowY': 'auto'},
                                                children=[
                                                    html.Small("No ROIs defined yet", className="text-muted")
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3"
                            ),

                            # Analysis Section
                            dbc.Card(
                                [
                                    dbc.CardHeader("Analysis"),
                                    dbc.CardBody(
                                        dbc.Button(
                                            "Analyze Current Frame",
                                            id='btn-analyze-frame',
                                            color='primary',
                                            className="w-100"
                                        ),
                                    ),
                                ],
                                className="mb-3"
                            ),
                        ],
                        md=3,
                        className="pe-3"
                    ),

                    # Right Panel: Display (75% width)
                    dbc.Col(
                        dbc.Tabs(
                            [
                                # Tab 1: Image Viewer
                                dbc.Tab(
                                    [
                                        dbc.Row(
                                            dbc.Col(
                                                [
                                                    dbc.ButtonGroup(
                                                        [
                                                            dbc.Button(
                                                                "◀ Prev",
                                                                id='btn-prev-frame',
                                                                color='secondary',
                                                                size='sm'
                                                            ),
                                                            dbc.Button(
                                                                "Next ▶",
                                                                id='btn-next-frame',
                                                                color='secondary',
                                                                size='sm'
                                                            ),
                                                        ],
                                                        className="me-3"
                                                    ),
                                                    html.Span(
                                                        id='frame-label',
                                                        className="ms-3 me-3 fw-bold"
                                                    ),
                                                    html.Span(
                                                        id='range-label',
                                                        className="me-3 text-muted small"
                                                    ),
                                                    html.Span(
                                                        id='saturation-label',
                                                        className="me-3 text-muted small"
                                                    ),
                                                ],
                                                className="mb-3 mt-3"
                                            )
                                        ),
                                        dcc.Graph(
                                            id='image-display',
                                            style={'height': '700px'},
                                            config={'responsive': True}
                                        ),
                                    ],
                                    label="Image Viewer"
                                ),

                                # Tab 2: Results
                                dbc.Tab(
                                    dcc.Textarea(
                                        id='results-text',
                                        readOnly=True,
                                        style={
                                            'width': '100%',
                                            'height': '700px',
                                            'fontFamily': 'monospace',
                                            'fontSize': '12px',
                                            'padding': '10px'
                                        },
                                        placeholder='Analysis results will appear here...'
                                    ),
                                    label="Results"
                                ),

                                # Tab 3: Histograms
                                dbc.Tab(
                                    dcc.Graph(
                                        id='histogram-plot',
                                        style={'height': '700px'},
                                        config={'responsive': True}
                                    ),
                                    label="Histograms"
                                ),
                            ],
                            id="tabs",
                            active_tab="tab-0"
                        ),
                        md=9
                    ),
                ]
            ),

            # Hidden stores for state management
            dcc.Store(id='data-store', data=None),
            dcc.Store(id='current-frame-idx', data=0),
            dcc.Store(id='rois-store', data=[]),
        ]
    )
