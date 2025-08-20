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
from core.attenuation_calc import AttenuationCalculator, ScanConditions, AttenuationSettings

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable objects"""
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

class ImageViewer(tk.Frame):
    """Image display widget with ROI selection capabilities"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.current_image = None
        self.display_image = None
        self.image_scale = 1.0
        self.rois = []
        self.roi_callback = None
        
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
        
        # Convert to 8-bit for display
        if image_array.dtype != np.uint8:
            # Use percentile scaling for better visibility
            p_low, p_high = np.percentile(image_array, [1, 99])
            image_scaled = np.clip((image_array - p_low) / (p_high - p_low) * 255, 0, 255).astype(np.uint8)
        else:
            image_scaled = image_array
        
        # Convert to PIL Image
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
        self.attenuation_calc = AttenuationCalculator()
        
        # Data storage
        self.current_frame_idx = 0
        self.analysis_results = None
        
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
        
        # Parameters section
        params_frame = ttk.LabelFrame(parent, text="Analysis Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Threshold
        ttk.Label(params_frame, text="Intensity Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.StringVar(value="0")
        ttk.Entry(params_frame, textvariable=self.threshold_var).pack(fill=tk.X, pady=2)
        
        # Saturation threshold
        ttk.Label(params_frame, text="Saturation Threshold:").pack(anchor=tk.W)
        self.sat_threshold_var = tk.StringVar(value="65535")
        ttk.Entry(params_frame, textvariable=self.sat_threshold_var).pack(fill=tk.X, pady=2)
        
        # Mask file
        ttk.Label(params_frame, text="Mask File (optional):").pack(anchor=tk.W)
        mask_frame = ttk.Frame(params_frame)
        mask_frame.pack(fill=tk.X, pady=2)
        self.mask_file_var = tk.StringVar()
        ttk.Entry(mask_frame, textvariable=self.mask_file_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(mask_frame, text="Browse", command=self.browse_mask_file).pack(side=tk.RIGHT)
        
        # Scan conditions
        scan_frame = ttk.LabelFrame(parent, text="Scan Conditions", padding=10)
        scan_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(scan_frame, text="X-ray Energy (keV):").pack(anchor=tk.W)
        self.energy_var = tk.StringVar(value="80.0")
        ttk.Entry(scan_frame, textvariable=self.energy_var).pack(fill=tk.X, pady=2)
        
        ttk.Label(scan_frame, text="Exposure Time (s):").pack(anchor=tk.W)
        self.exposure_var = tk.StringVar(value="0.5")
        ttk.Entry(scan_frame, textvariable=self.exposure_var).pack(fill=tk.X, pady=2)
        
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
        
        ttk.Button(analysis_frame, text="Run Analysis", 
                  command=self.run_analysis).pack(fill=tk.X, pady=2)
        ttk.Button(analysis_frame, text="Export Results", 
                  command=self.export_results).pack(fill=tk.X, pady=2)
        
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
                self.log_message(f"Successfully loaded {self.data_handler.shape[0]} frames")
                self.current_frame_idx = 0
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
                self.log_message(f"Successfully loaded {self.data_handler.shape[0]} images")
                self.current_frame_idx = 0
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
    
    def update_display(self):
        """Update the image display"""
        if self.data_handler.data is not None:
            frame = self.data_handler.get_frame(self.current_frame_idx)
            self.image_viewer.set_image(frame)
            
            total_frames = self.data_handler.shape[0]
            self.frame_label.config(text=f"Frame: {self.current_frame_idx+1}/{total_frames}")
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.data_handler.data is not None and self.current_frame_idx > 0:
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
        """Run the complete analysis"""
        if self.data_handler.data is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
        
        self.log_message("Starting analysis...")
        
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
        
        # Run analysis in separate thread
        threading.Thread(target=self._run_analysis_thread, daemon=True).start()
    
    def _run_analysis_thread(self):
        """Run analysis in background thread"""
        try:
            # Generate analysis report
            self.analysis_results = self.analysis_engine.generate_report(self.data_handler.data)
            
            # Calculate attenuation report
            energy = float(self.energy_var.get())
            exposure = float(self.exposure_var.get())
            
            scan_conditions = ScanConditions(
                energy_kev=energy,
                exposure_time_s=exposure,
                attenuation_settings=[]  # Would be populated from GUI inputs
            )
            
            overall_stats = self.analysis_results['overall_statistics'].get('overall', {})
            attenuation_report = self.attenuation_calc.generate_attenuation_report(
                scan_conditions, overall_stats
            )
            
            self.analysis_results['attenuation_analysis'] = attenuation_report
            
            # Update GUI on main thread
            self.root.after(0, self._display_results)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {e}"))
    
    def _display_results(self):
        """Display analysis results"""
        if not self.analysis_results:
            return
        
        self.log_message("Analysis complete!")
        
        # Display text results
        results_text = json.dumps(self.analysis_results, indent=2, cls=CustomJSONEncoder)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, results_text)
        
        # Plot histograms
        self._plot_histograms()
    
    def _plot_histograms(self):
        """Plot histograms for overall and ROI data"""
        if not self.analysis_results or 'histograms' not in self.analysis_results:
            return
        
        # Clear previous plots
        for ax in self.axes.flat:
            ax.clear()
        
        histograms = self.analysis_results['histograms']
        
        # Plot overall histogram
        if 'overall' in histograms:
            hist_data = histograms['overall']
            self.axes[0, 0].bar(hist_data['bins'], hist_data['counts'])
            self.axes[0, 0].set_title('Overall Histogram')
            self.axes[0, 0].set_xlabel('Intensity')
            self.axes[0, 0].set_ylabel('Counts')
        
        # Plot ROI histograms (up to 3)
        roi_names = [name for name in histograms.keys() if name != 'overall']
        for i, roi_name in enumerate(roi_names[:3]):
            row = i // 2
            col = (i + 1) % 2
            if row < 2:
                hist_data = histograms[roi_name]
                self.axes[row, col].bar(hist_data['bins'], hist_data['counts'])
                self.axes[row, col].set_title(f'ROI: {roi_name}')
                self.axes[row, col].set_xlabel('Intensity')
                self.axes[row, col].set_ylabel('Counts')
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def export_results(self):
        """Export analysis results"""
        if not self.analysis_results:
            messagebox.showwarning("Warning", "No results to export. Run analysis first.")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    json.dump(self.analysis_results, f, indent=2, cls=CustomJSONEncoder)
                self.log_message(f"Results exported to: {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {e}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = HEDMAnalyzer(root)
    root.mainloop()

if __name__ == "__main__":
    main()