"""
Main GUI window for HEDM X-ray image analysis
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import os
import threading
import logging

# Import core modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data_handler import DataHandler
from core.analysis_engine import AnalysisEngine, ROI, Statistics
from core.parameter_optimizer import ParameterOptimizer

class ImageViewer(tk.Frame):
    """Image display widget with ROI selection capabilities"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.current_image = None
        self.display_image = None
        self.image_scale = 1.0
        self.rois = []
        self.roi_callback = None
        self.saturation_threshold = 65535  # Default for 16-bit detectors
        self.highlight_saturated = True
        
        # Create canvas for image display
        self.canvas = tk.Canvas(self, bg='black', width=600, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for ROI selection
        self.canvas.bind("<Button-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        
        self.roi_start = None
        self.roi_rect = None
    
    def set_image(self, image_array: np.ndarray):
        """Display image array on canvas"""
        if image_array is None:
            return
        
        self.current_image = image_array
        
        # Convert to RGB for display and highlight saturated pixels
        if image_array.dtype != np.uint8:
            # Use percentile scaling for better visibility
            p_low, p_high = np.percentile(image_array, [1, 99])
            image_scaled = np.clip((image_array - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        else:
            image_scaled = image_array
        
        # Create RGB image for saturation highlighting
        if self.highlight_saturated:
            # Find saturated pixels in original image
            saturated_mask = image_array >= self.saturation_threshold
            
            # Convert grayscale to RGB
            rgb_image = np.stack([image_scaled, image_scaled, image_scaled], axis=-1)
            
            # Highlight saturated pixels in red
            rgb_image[saturated_mask] = [255, 0, 0]  # Pure red
            
            pil_image = Image.fromarray(rgb_image)
        else:
            pil_image = Image.fromarray(image_scaled)
        
        # Scale to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:  # Canvas has been rendered
            img_width, img_height = pil_image.size
            
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.image_scale = min(scale_x, scale_y, 1.0)  # Don't upscale
            
            new_width = int(img_width * self.image_scale)
            new_height = int(img_height * self.image_scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.display_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.canvas.winfo_width()//2, 
            self.canvas.winfo_height()//2, 
            image=self.display_image, 
            anchor=tk.CENTER
        )
        
        # Redraw ROIs
        self.draw_rois()
    
    def on_mouse_press(self, event):
        """Start ROI selection"""
        self.roi_start = (event.x, event.y)
    
    def on_mouse_drag(self, event):
        """Update ROI rectangle during drag"""
        if self.roi_start:
            if self.roi_rect:
                self.canvas.delete(self.roi_rect)
            
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            
            self.roi_rect = self.canvas.create_rectangle(
                x1, y1, x2, y2, outline='red', width=2
            )
    
    def on_mouse_release(self, event):
        """Finish ROI selection"""
        if self.roi_start and self.roi_callback:
            x1, y1 = self.roi_start
            x2, y2 = event.x, event.y
            
            # Convert canvas coordinates to image coordinates
            if self.current_image is not None:
                img_coords = self.canvas_to_image_coords(x1, y1, x2, y2)
                if img_coords:
                    self.roi_callback(img_coords)
        
        self.roi_start = None
        if self.roi_rect:
            self.canvas.delete(self.roi_rect)
            self.roi_rect = None
    
    def canvas_to_image_coords(self, x1, y1, x2, y2):
        """Convert canvas coordinates to image pixel coordinates"""
        if not self.display_image:
            return None
        
        # Get image position on canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width = self.display_image.width()
        img_height = self.display_image.height()
        
        # Image is centered
        img_x = (canvas_width - img_width) // 2
        img_y = (canvas_height - img_height) // 2
        
        # Convert to image coordinates
        img_x1 = int((x1 - img_x) / self.image_scale)
        img_y1 = int((y1 - img_y) / self.image_scale)
        img_x2 = int((x2 - img_x) / self.image_scale)
        img_y2 = int((y2 - img_y) / self.image_scale)
        
        # Ensure proper ordering
        img_x1, img_x2 = min(img_x1, img_x2), max(img_x1, img_x2)
        img_y1, img_y2 = min(img_y1, img_y2), max(img_y1, img_y2)
        
        # Clamp to image bounds
        img_height_orig, img_width_orig = self.current_image.shape
        img_x1 = max(0, min(img_x1, img_width_orig-1))
        img_x2 = max(0, min(img_x2, img_width_orig-1))
        img_y1 = max(0, min(img_y1, img_height_orig-1))
        img_y2 = max(0, min(img_y2, img_height_orig-1))
        
        return (img_x1, img_y1, img_x2, img_y2)
    
    def set_roi_callback(self, callback):
        """Set callback function for ROI selection"""
        self.roi_callback = callback
    
    def set_saturation_threshold(self, threshold):
        """Set saturation threshold for highlighting"""
        self.saturation_threshold = threshold
        # Refresh display if image is loaded
        if self.current_image is not None:
            self.set_image(self.current_image)
    
    def set_saturation_highlighting(self, enabled):
        """Enable/disable saturation highlighting"""
        self.highlight_saturated = enabled
        # Refresh display if image is loaded
        if self.current_image is not None:
            self.set_image(self.current_image)
    
    def draw_rois(self):
        """Draw ROI overlays on image"""
        # This would be implemented to show existing ROIs
        pass

class HEDMAnalyzer:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HEDM X-ray Image Analyzer")
        self.root.geometry("1400x900")

        # Initialize core components
        self.data_handler = DataHandler()
        self.analysis_engine = AnalysisEngine()
        self.parameter_optimizer = ParameterOptimizer(logger=logging.getLogger(__name__))

        # Data storage
        self.current_frame_idx = 0
        self.skip_frames = 0  # Number of frames to skip at the beginning
        self.analysis_results = None
        self.param_results = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.setup_gui()
    
    def setup_gui(self):
        """Setup the main GUI layout"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_paned)
        main_paned.add(left_panel, weight=1)
        
        # Right panel for image and plots
        right_panel = ttk.Frame(main_paned)
        main_paned.add(right_panel, weight=3)
        
        self.setup_left_panel(left_panel)
        self.setup_right_panel(right_panel)
    
    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # File input section
        file_frame = ttk.LabelFrame(parent, text="Data Input", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Load HDF5 File",
                  command=self.load_hdf5_file).pack(fill=tk.X, pady=2)
        ttk.Button(file_frame, text="Load Image Sequence",
                  command=self.load_image_sequence).pack(fill=tk.X, pady=2)

        # Skip frames control
        skip_frame = ttk.Frame(file_frame)
        skip_frame.pack(fill=tk.X, pady=5)
        ttk.Label(skip_frame, text="Skip Frames:").pack(side=tk.LEFT)
        self.skip_frames_var = tk.IntVar(value=0)
        skip_spinbox = ttk.Spinbox(skip_frame, from_=0, to=10,
                                    textvariable=self.skip_frames_var,
                                    width=10, command=self.on_skip_frames_change)
        skip_spinbox.pack(side=tk.LEFT, padx=5)
        skip_spinbox.bind('<KeyRelease>', lambda e: self.on_skip_frames_change())
        ttk.Label(skip_frame, text="(Start from frame N+1)").pack(side=tk.LEFT)
        
        # Parameters section
        params_frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Saturation threshold
        ttk.Label(params_frame, text="Saturation Threshold:").pack(anchor=tk.W)
        self.sat_threshold_var = tk.StringVar(value="65535")
        sat_entry = ttk.Entry(params_frame, textvariable=self.sat_threshold_var)
        sat_entry.pack(fill=tk.X, pady=2)
        sat_entry.bind('<KeyRelease>', self.on_saturation_threshold_change)
        
        # Saturation highlighting toggle
        self.highlight_saturation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params_frame, text="Highlight Saturated Pixels (Red)", 
                       variable=self.highlight_saturation_var,
                       command=self.on_highlight_toggle).pack(anchor=tk.W, pady=2)
        
        # Mask file
        ttk.Label(params_frame, text="Mask File (optional):").pack(anchor=tk.W)
        mask_frame = ttk.Frame(params_frame)
        mask_frame.pack(fill=tk.X, pady=2)
        self.mask_file_var = tk.StringVar()
        ttk.Entry(mask_frame, textvariable=self.mask_file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(mask_frame, text="Browse", command=self.browse_mask_file).pack(side=tk.RIGHT)

        # Parameter Recommendation section
        param_rec_frame = ttk.LabelFrame(parent, text="Parameter Recommendation", padding=10)
        param_rec_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(param_rec_frame, text="Experiment Data Directory:").pack(anchor=tk.W)
        param_dir_frame = ttk.Frame(param_rec_frame)
        param_dir_frame.pack(fill=tk.X, pady=2)
        self.param_dir_var = tk.StringVar(value="./experiment_data")
        ttk.Entry(param_dir_frame, textvariable=self.param_dir_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(param_dir_frame, text="Browse", command=self.browse_param_directory).pack(side=tk.RIGHT, padx=2)

        ttk.Button(param_rec_frame, text="Get Parameters",
                  command=self.get_parameters).pack(fill=tk.X, pady=2)

        ttk.Label(param_rec_frame, text="Analyzes folders: sample_atten-X_t-Y",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor=tk.W, pady=(0, 5))

        # ROI management
        roi_frame = ttk.LabelFrame(parent, text="ROI Management", padding=10)
        roi_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(roi_frame, text="ROI Name:").pack(anchor=tk.W)
        self.roi_name_var = tk.StringVar(value="ROI_1")
        ttk.Entry(roi_frame, textvariable=self.roi_name_var).pack(fill=tk.X, pady=2)
        
        ttk.Button(roi_frame, text="Enable ROI Selection", 
                  command=self.enable_roi_selection).pack(fill=tk.X, pady=2)
        ttk.Button(roi_frame, text="Clear All ROIs", 
                  command=self.clear_rois).pack(fill=tk.X, pady=2)
        
        # ROI list
        self.roi_listbox = tk.Listbox(roi_frame, height=4)
        self.roi_listbox.pack(fill=tk.X, pady=2)
        
        # Analysis controls
        analysis_frame = ttk.LabelFrame(parent, text="Analysis", padding=10)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(analysis_frame, text="Analyze Current Frame",
                  command=self.run_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Analyze All Frames",
                  command=self.run_all_frames_analysis).pack(fill=tk.X, pady=2)

        # Lower threshold
        ttk.Separator(analysis_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Label(analysis_frame, text="Lower Threshold:").pack(anchor=tk.W)
        threshold_frame = ttk.Frame(analysis_frame)
        threshold_frame.pack(fill=tk.X, pady=2)
        self.threshold_var = tk.StringVar(value="0")
        ttk.Entry(threshold_frame, textvariable=self.threshold_var).pack(fill=tk.X)
        ttk.Label(analysis_frame, text="(Exclude pixels below this value from analysis)",
                 font=('TkDefaultFont', 8), foreground='gray').pack(anchor=tk.W, pady=(0, 5))

        # Status log
        log_frame = ttk.LabelFrame(parent, text="Status Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def setup_right_panel(self, parent):
        """Setup the right panel with image viewer and plots"""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image viewer tab
        image_frame = ttk.Frame(notebook)
        notebook.add(image_frame, text="Image Viewer")
        
        # Frame navigation
        nav_frame = ttk.Frame(image_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(nav_frame, text="◀ Prev", command=self.prev_frame).pack(side=tk.LEFT)
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="Next ▶", command=self.next_frame).pack(side=tk.LEFT)
        
        # Dynamic range display
        self.range_label = ttk.Label(nav_frame, text="Range: - / -")
        self.range_label.pack(side=tk.RIGHT, padx=10)

        # Saturated pixels display
        self.saturation_label = ttk.Label(nav_frame, text="Saturated: - / -")
        self.saturation_label.pack(side=tk.RIGHT, padx=10)
        
        # Image viewer
        self.image_viewer = ImageViewer(image_frame)
        self.image_viewer.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_viewer.set_roi_callback(self.on_roi_selected)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        # Results will be displayed as text and plots
        self.results_text = scrolledtext.ScrolledText(results_frame, font=('Courier', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plots tab
        plots_frame = ttk.Frame(notebook)
        notebook.add(plots_frame, text="Histograms")
        
        # Matplotlib figure for histograms
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, plots_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def log_message(self, message):
        """Add message to status log"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def load_hdf5_file(self):
        """Load HDF5 file dialog"""
        filename = filedialog.askopenfilename(
            title="Select HDF5 File",
            filetypes=[("HDF5 files", "*.h5 *.hdf5"), ("All files", "*.*")]
        )

        if filename:
            self.log_message(f"Loading HDF5 file: {filename}")
            success = self.data_handler.load_hdf5(filename)

            if success:
                total_frames = self.data_handler.shape[0]
                self.log_message(f"Successfully loaded {total_frames} frames")

                # Set skip_frames from UI and validate
                self.skip_frames = self.skip_frames_var.get()
                if self.skip_frames >= total_frames:
                    self.skip_frames = 0
                    self.skip_frames_var.set(0)
                    self.log_message(f"Reset skip frames to 0 (dataset has {total_frames} frames)")

                # Start at first non-skipped frame
                self.current_frame_idx = self.skip_frames
                if self.skip_frames > 0:
                    self.log_message(f"Skipping first {self.skip_frames} frame(s), starting from frame {self.skip_frames + 1}")

                self.update_display()
            else:
                messagebox.showerror("Error", "Failed to load HDF5 file")
    
    def load_image_sequence(self):
        """Load image sequence dialog"""
        directory = filedialog.askdirectory(title="Select Image Directory")

        if directory:
            self.log_message(f"Loading image sequence from: {directory}")
            success = self.data_handler.load_image_sequence(directory, "*.png")

            if success:
                total_frames = self.data_handler.shape[0]
                self.log_message(f"Successfully loaded {total_frames} images")

                # Set skip_frames from UI and validate
                self.skip_frames = self.skip_frames_var.get()
                if self.skip_frames >= total_frames:
                    self.skip_frames = 0
                    self.skip_frames_var.set(0)
                    self.log_message(f"Reset skip frames to 0 (dataset has {total_frames} frames)")

                # Start at first non-skipped frame
                self.current_frame_idx = self.skip_frames
                if self.skip_frames > 0:
                    self.log_message(f"Skipping first {self.skip_frames} frame(s), starting from frame {self.skip_frames + 1}")

                self.update_display()
            else:
                messagebox.showerror("Error", "Failed to load image sequence")
    
    def browse_mask_file(self):
        """Browse for mask file"""
        filename = filedialog.askopenfilename(
            title="Select Mask File",
            filetypes=[("Image files", "*.png *.tiff *.tif"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filename:
            self.mask_file_var.set(filename)

    def on_skip_frames_change(self):
        """Handle skip frames value change"""
        try:
            skip_val = self.skip_frames_var.get()
            if self.data_handler.data is not None:
                total_frames = self.data_handler.shape[0]
                if skip_val >= total_frames:
                    messagebox.showwarning("Warning",
                        f"Cannot skip {skip_val} frames - dataset only has {total_frames} frames.\n" +
                        f"Maximum skip value is {total_frames - 1}")
                    self.skip_frames_var.set(max(0, total_frames - 1))
                    skip_val = self.skip_frames_var.get()

                self.skip_frames = skip_val

                # Reset to first valid frame if current frame is now skipped
                if self.current_frame_idx < self.skip_frames:
                    self.current_frame_idx = self.skip_frames
                    self.update_display()

                self.log_message(f"Skip frames set to {skip_val}. Analysis will start from frame {skip_val + 1}")
            else:
                self.skip_frames = skip_val
        except (ValueError, tk.TclError):
            pass  # Invalid input, ignore
    
    def on_saturation_threshold_change(self, event=None):
        """Handle saturation threshold changes"""
        try:
            threshold = int(self.sat_threshold_var.get())
            self.image_viewer.set_saturation_threshold(threshold)
            # Update saturated pixel display
            self.update_saturated_pixel_display()
        except ValueError:
            pass  # Invalid input, ignore
    
    def on_highlight_toggle(self):
        """Handle saturation highlighting toggle"""
        enabled = self.highlight_saturation_var.get()
        self.image_viewer.set_saturation_highlighting(enabled)

    def update_saturated_pixel_display(self):
        """Update saturated pixels display based on current frame and threshold"""
        if self.data_handler.data is not None:
            try:
                frame = self.data_handler.get_frame(self.current_frame_idx)
                sat_threshold = int(self.sat_threshold_var.get())
                saturated_mask = frame >= sat_threshold
                saturated_count = int(np.sum(saturated_mask))
                total_pixels = frame.size
                saturated_percentage = (saturated_count / total_pixels) * 100
                self.saturation_label.config(text=f"Saturated: {saturated_count} ({saturated_percentage:.2f}%)")
            except (ValueError, IndexError):
                self.saturation_label.config(text="Saturated: - / -")
    
    def update_display(self):
        """Update the image display"""
        if self.data_handler.data is not None:
            frame = self.data_handler.get_frame(self.current_frame_idx)
            
            # Update saturation threshold in image viewer
            try:
                sat_threshold = int(self.sat_threshold_var.get())
                self.image_viewer.set_saturation_threshold(sat_threshold)
            except ValueError:
                pass  # Invalid threshold, keep current value
            
            self.image_viewer.set_image(frame)
            
            total_frames = self.data_handler.shape[0]
            self.frame_label.config(text=f"Frame: {self.current_frame_idx+1}/{total_frames}")
            
            # Update dynamic range display
            min_val = int(np.min(frame))
            max_val = int(np.max(frame))
            self.range_label.config(text=f"Range: {min_val} / {max_val}")

            # Update saturated pixels display
            self.update_saturated_pixel_display()
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.data_handler.data is not None and self.current_frame_idx > self.skip_frames:
            self.current_frame_idx -= 1
            self.update_display()
    
    def next_frame(self):
        """Go to next frame"""
        if self.data_handler.data is not None:
            max_frame = self.data_handler.shape[0] - 1
            if self.current_frame_idx < max_frame:
                self.current_frame_idx += 1
                self.update_display()
    
    def enable_roi_selection(self):
        """Enable ROI selection mode"""
        if self.data_handler.data is not None:
            self.log_message("ROI selection enabled. Click and drag on image to select region.")
        else:
            messagebox.showwarning("Warning", "Please load data first")
    
    def on_roi_selected(self, roi_coords):
        """Handle ROI selection"""
        x1, y1, x2, y2 = roi_coords
        roi_name = self.roi_name_var.get()
        
        if not roi_name:
            roi_name = f"ROI_{len(self.analysis_engine.rois)+1}"
        
        roi = ROI(roi_name, x1, y1, x2, y2)
        self.analysis_engine.add_roi(roi)
        
        # Update ROI list
        self.roi_listbox.insert(tk.END, f"{roi_name}: ({x1},{y1})-({x2},{y2})")
        
        self.log_message(f"Added ROI: {roi_name} at {roi_coords}")
    
    def clear_rois(self):
        """Clear all ROIs"""
        self.analysis_engine.clear_rois()
        self.roi_listbox.delete(0, tk.END)
        self.log_message("Cleared all ROIs")
    
    def run_analysis(self):
        """Run analysis on current frame only"""
        if self.data_handler.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        self.log_message(f"Analyzing frame {self.current_frame_idx+1}...")

        # Set analysis parameters
        try:
            threshold = float(self.threshold_var.get())
            sat_threshold = int(self.sat_threshold_var.get())

            self.analysis_engine.set_threshold(threshold)
            self.analysis_engine.set_saturation_threshold(sat_threshold)
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values")
            return

        # Load mask if specified
        mask_file = self.mask_file_var.get()
        if mask_file and os.path.exists(mask_file):
            mask = self.data_handler.load_mask_file(mask_file)
            if mask is not None:
                self.analysis_engine.set_mask(mask)

        # Get current frame
        frame = self.data_handler.get_frame(self.current_frame_idx)

        # Analyze frame (fast - no threading needed)
        frame_stats = self.analysis_engine.analyze_frame(frame)
        saturation_stats = self.analysis_engine.calculate_saturation_analysis(frame)

        # Store results
        self.analysis_results = {
            'frame_index': self.current_frame_idx,
            'statistics': {name: stats.to_dict() for name, stats in frame_stats.items()},
            'saturation': saturation_stats
        }

        # Display results
        self._display_results(frame)

    def run_all_frames_analysis(self):
        """Run analysis on all frames and save statistics to file"""
        if self.data_handler.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return

        # Ask user where to save the report
        filename = filedialog.asksaveasfilename(
            title="Save All Frames Analysis Report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if not filename:
            return  # User cancelled

        total_frames = self.data_handler.shape[0]
        frames_to_analyze = total_frames - self.skip_frames

        self.log_message(f"Starting all frames analysis...")
        if self.skip_frames > 0:
            self.log_message(f"Skipping first {self.skip_frames} frame(s)")
            self.log_message(f"Analyzing {frames_to_analyze} frames (frame {self.skip_frames + 1} to {total_frames})")
        else:
            self.log_message(f"Analyzing {frames_to_analyze} frames")
        self.log_message("This may take a while depending on dataset size...")

        # Set analysis parameters
        try:
            threshold = float(self.threshold_var.get())
            sat_threshold = int(self.sat_threshold_var.get())

            self.analysis_engine.set_threshold(threshold)
            self.analysis_engine.set_saturation_threshold(sat_threshold)
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter values")
            return

        # Load mask if specified
        mask_file = self.mask_file_var.get()
        if mask_file and os.path.exists(mask_file):
            mask = self.data_handler.load_mask_file(mask_file)
            if mask is not None:
                self.analysis_engine.set_mask(mask)

        # Run analysis in separate thread (this takes longer)
        threading.Thread(
            target=self._run_all_frames_analysis_thread,
            args=(filename,),
            daemon=True
        ).start()

    def _run_all_frames_analysis_thread(self, output_filename):
        """Run all frames analysis in background thread"""
        try:
            # Analyze all frames (skipping the first N frames)
            results = self.analysis_engine.analyze_all_frames_stats(
                self.data_handler.data,
                skip_frames=self.skip_frames
            )

            # Add timestamp and skip info
            from datetime import datetime
            results['analysis_timestamp'] = datetime.now().isoformat()
            results['skip_frames'] = self.skip_frames

            # Save to file
            with open(output_filename, 'w') as f:
                json.dump(results, f, indent=2)

            # Update GUI on main thread
            self.root.after(0, lambda: self._all_frames_analysis_complete(output_filename))

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"All frames analysis failed: {e}"))
            self.root.after(0, lambda: self.log_message(f"ERROR: {e}"))

    def _all_frames_analysis_complete(self, output_filename):
        """Called when all frames analysis completes"""
        self.log_message("=" * 60)
        self.log_message("All frames analysis complete!")
        self.log_message(f"Report saved to: {output_filename}")
        self.log_message("=" * 60)
        messagebox.showinfo("Success", f"All frames analysis complete!\n\nReport saved to:\n{output_filename}")

    def _display_results(self, frame):
        """Display analysis results"""
        if not self.analysis_results:
            return

        self.log_message("Analysis complete!")

        # Display text results
        results_text = json.dumps(self.analysis_results, indent=2)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)

        # Plot histograms
        self._plot_histograms(frame)
    
    def _plot_histograms(self, frame):
        """Plot histogram PDFs for overall frame and ROIs"""
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()

        # Get histogram data for overall frame
        overall_data = self.analysis_engine.calculate_histogram(frame)

        if len(overall_data) > 0:
            # Plot overall histogram as PDF
            self.axes[0, 0].hist(overall_data, bins='auto', density=True, alpha=0.7, color='blue', edgecolor='black')
            self.axes[0, 0].set_title('Overall Frame - PDF')
            self.axes[0, 0].set_xlabel('Intensity')
            self.axes[0, 0].set_ylabel('Probability Density')
            self.axes[0, 0].grid(True, alpha=0.3)

        # Plot ROI histograms (up to 3)
        roi_positions = [(0, 1), (1, 0), (1, 1)]  # Positions for up to 3 ROIs
        for i, roi in enumerate(self.analysis_engine.rois[:3]):
            if i < len(roi_positions):
                row, col = roi_positions[i]
                roi_data = self.analysis_engine.calculate_histogram(frame, roi=roi)

                if len(roi_data) > 0:
                    self.axes[row, col].hist(roi_data, bins='auto', density=True, alpha=0.7, color='green', edgecolor='black')
                    self.axes[row, col].set_title(f'ROI: {roi.name} - PDF')
                    self.axes[row, col].set_xlabel('Intensity')
                    self.axes[row, col].set_ylabel('Probability Density')
                    self.axes[row, col].grid(True, alpha=0.3)
                else:
                    self.axes[row, col].text(0.5, 0.5, 'No data', ha='center', va='center', transform=self.axes[row, col].transAxes)
                    self.axes[row, col].set_title(f'ROI: {roi.name}')

        # Hide unused subplots
        num_rois = len(self.analysis_engine.rois)
        if num_rois < 3:
            for i in range(num_rois, 3):
                row, col = roi_positions[i]
                self.axes[row, col].axis('off')

        self.fig.tight_layout()
        self.canvas.draw_idle()  # Use draw_idle() for better Tkinter compatibility

    def browse_param_directory(self):
        """Browse for parameter recommendation directory"""
        directory = filedialog.askdirectory(title="Select Experiment Data Directory")
        if directory:
            self.param_dir_var.set(directory)
            self.log_message(f"Parameter directory set to: {directory}")

    def get_parameters(self):
        """Get parameter recommendations from experiment data"""
        directory = self.param_dir_var.get()

        if not directory:
            messagebox.showerror("Error", "No directory specified")
            return

        if not os.path.isdir(directory):
            messagebox.showerror("Error", f"Directory not found: {directory}")
            return

        # Run analysis in a separate thread to prevent UI blocking
        thread = threading.Thread(target=self._analyze_parameters_thread, args=(directory,))
        thread.daemon = True
        thread.start()

    def _analyze_parameters_thread(self, directory):
        """Background thread for parameter analysis"""
        try:
            self.log_message(f"Analyzing parameter directory: {directory}")
            results = self.parameter_optimizer.analyze_directory(directory)

            if not results['success']:
                self.root.after(0, lambda: messagebox.showerror("Analysis Error", f"Error: {results['error']}"))
                self.log_message(f"Analysis failed: {results['error']}")
                return

            self.param_results = results
            self.log_message(f"Analysis complete. Found {len(results['predictions'])} predictions.")
            # Schedule GUI update on main thread
            self.root.after(0, self.show_parameter_results)

        except Exception as e:
            self.log_message(f"Error during analysis: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            import traceback
            traceback.print_exc()

    def show_parameter_results(self):
        """Display parameter recommendations in a new window"""
        if not self.param_results or not self.param_results.get('success'):
            messagebox.showerror("Error", "No valid results to display")
            return

        # Create new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("Parameter Recommendations")
        results_window.geometry("1000x700")

        # Create notebook for tabs
        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Summary tab
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")

        fit = self.param_results['fit_results']
        summary_text = f"""FITTED PARAMETERS
{'='*60}

Attenuation Coefficient (μ):     {fit['mu']:.6f} mm⁻¹
Unattenuated Flux (I₀):          {fit['I0']:.6e} counts/s
R² (Goodness of Fit):            {fit['r_squared']:.6f}

{'='*60}

This data was fitted using the Beer-Lambert law:
    I(x) = I₀ × exp(-μ × x)

where:
    I(x) = intensity rate at thickness x
    x = material thickness (mm)
    μ = attenuation coefficient
    I₀ = unattenuated flux
"""

        summary_text_widget = scrolledtext.ScrolledText(summary_frame, font=('Courier', 10))
        summary_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        summary_text_widget.insert(tk.END, summary_text)
        summary_text_widget.config(state=tk.DISABLED)

        # Table tab
        table_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Recommendations")

        # Create table with predictions
        table_text = f"""{'Atten Idx':<12} | {'Thickness':<12} | {'Predicted Rate':<20} | {'Max Time (s)':<15}
{'-'*65}
"""
        for p in self.param_results['predictions']:
            max_time_str = f"{p['max_time']:.6f}" if p['max_time'] != float('inf') else "∞"
            table_text += f"atten-{p['atten_idx']:<6} | {p['thickness']:<12.2f} | {p['rate']:<20.6e} | {max_time_str:<15}\n"

        table_text_widget = scrolledtext.ScrolledText(table_frame, font=('Courier', 9))
        table_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        table_text_widget.insert(tk.END, table_text)
        table_text_widget.config(state=tk.DISABLED)

        # Plot tab
        plot_frame = ttk.Frame(notebook)
        notebook.add(plot_frame, text="Regression Plot")

        self._create_regression_plot(plot_frame)

        # Button to close
        close_button = ttk.Button(results_window, text="Close", command=results_window.destroy)
        close_button.pack(pady=5)

    def _create_regression_plot(self, parent):
        """Create and display regression plot"""
        try:
            data_points = self.param_results['data_points']
            fit = self.param_results['fit_results']

            # Scatter data (actual measurements)
            thicknesses = sorted(list(data_points.keys()))
            avg_rates = [np.mean(data_points[t]) for t in thicknesses]

            # Use Tkinter backend to avoid threading issues on macOS
            plt.switch_backend('TkAgg')

            # Create figure with explicit backend
            fig, ax = plt.subplots(figsize=(8, 6), dpi=80)
            fig.patch.set_facecolor('white')

            # Plot scatter points
            ax.scatter(thicknesses, avg_rates, color='blue', s=100, label='Measured Data', zorder=3)

            # Regression line
            if thicknesses:
                x_fit = np.linspace(0, max(thicknesses) * 1.1, 100)
                y_fit = fit['I0'] * np.exp(-fit['mu'] * x_fit)
                ax.plot(x_fit, y_fit, 'r--', label='Fitted Curve', linewidth=2)

            ax.set_xlabel('Cu Thickness (mm)', fontsize=12)
            ax.set_ylabel('Intensity Rate (counts/s)', fontsize=12)
            ax.set_title('Beer-Lambert Law Fit', fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
            fig.tight_layout()

            # Embed in Tkinter - this MUST be on main thread
            canvas = FigureCanvasTkAgg(fig, parent)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            canvas.draw()

        except Exception as e:
            error_label = ttk.Label(parent, text=f"Error creating plot: {str(e)}", foreground='red')
            error_label.pack(padx=5, pady=5)


def main():
    """Main application entry point"""
    root = tk.Tk()
    app = HEDMAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()