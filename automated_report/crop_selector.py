"""
Interactive crop selection tool using napari.

Uses a Points layer for clicking cell centers and a Shapes layer
for real-time preview of crop boxes.
"""

import numpy as np
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import copy

# Lazy import napari (only needed for interactive selection)
napari = None

def _ensure_napari():
    """Import napari on demand."""
    global napari
    if napari is None:
        import napari as _napari
        napari = _napari
    return napari


@dataclass
class ChannelContrast:
    """Contrast limits for a single channel."""
    min_val: float
    max_val: float
    auto_percentile: Optional[tuple[float, float]] = None  # e.g., (0.1, 99.9)
    
    @classmethod
    def from_data(cls, data: np.ndarray, percentile: tuple[float, float] = (0.1, 99.9)):
        """Calculate contrast limits from data using percentiles."""
        min_val = float(np.percentile(data, percentile[0]))
        max_val = float(np.percentile(data, percentile[1]))
        return cls(min_val=min_val, max_val=max_val, auto_percentile=percentile)


@dataclass
class CropMetadata:
    """Complete metadata for a single crop."""
    # Source information
    source_file: str
    source_name: str
    
    # Crop location (in pixels, relative to original image)
    center_pixel: tuple[int, int]  # (y, x)
    box_size_pixels: tuple[int, int]  # (height, width)
    
    # Physical units
    center_physical_um: tuple[float, float]  # (y, x) in microns
    box_size_physical_um: float  # Side length in microns
    pixel_size_um: tuple[float, float]  # (y, x) pixel sizes
    
    # Z information
    z_range: tuple[int, int]  # (start, end) z-slices used
    z_projection: str = "max"  # Projection method
    
    # Contrast settings per channel
    channel_contrasts: list[ChannelContrast] = field(default_factory=list)
    
    # Channel info from source
    channel_wavelengths: list[dict] = field(default_factory=list)
    channel_colors_original: list[str] = field(default_factory=list)
    
    # Which channels to include in figures (indices into original channel list)
    # If None or empty, all channels are included
    visible_channels: Optional[list[int]] = None
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""
    crop_id: str = ""
    
    def get_active_channels(self) -> list[int]:
        """Get list of channel indices to include in figures."""
        if self.visible_channels is None or len(self.visible_channels) == 0:
            # Default: all channels
            result = list(range(len(self.channel_contrasts)))
        else:
            result = self.visible_channels
        return result
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Convert ChannelContrast objects
        d['channel_contrasts'] = [asdict(cc) for cc in self.channel_contrasts]
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'CropMetadata':
        """Reconstruct from dictionary."""
        d = copy.deepcopy(d)
        d['channel_contrasts'] = [ChannelContrast(**cc) for cc in d['channel_contrasts']]
        # Convert lists back to tuples
        d['center_pixel'] = tuple(d['center_pixel'])
        d['box_size_pixels'] = tuple(d['box_size_pixels'])
        d['center_physical_um'] = tuple(d['center_physical_um'])
        d['pixel_size_um'] = tuple(d['pixel_size_um'])
        d['z_range'] = tuple(d['z_range'])
        return cls(**d)


class CropSelector:
    """
    Interactive napari-based tool for selecting cell crops.
    
    Usage:
        selector = CropSelector(czi_image, box_size_um=50.0)
        selector.run()  # Opens napari window
        # Click on cells to add crop points
        # Points can be moved/deleted
        # Close napari when done
        selector.save_session("my_crops.json")
    """
    
    def __init__(self, czi_image, box_size_um: float = 50.0, z_project_for_display: bool = True):
        """
        Initialize the crop selector.
        
        Args:
            czi_image: CziImage instance with loaded data
            box_size_um: Side length of crop box in microns
            z_project_for_display: If True, show max projection for easier selection
        """
        self.czi_image = czi_image
        self.box_size_um = box_size_um
        self.z_project_for_display = z_project_for_display
        
        # Get pixel sizes from metadata
        self.pixel_size_x_um = czi_image.scale.get('X', 1.0) * 1e6  # Convert m to um
        self.pixel_size_y_um = czi_image.scale.get('Y', 1.0) * 1e6
        
        # Calculate box size in pixels
        self.box_width_px = int(np.ceil(box_size_um / self.pixel_size_x_um))
        self.box_height_px = int(np.ceil(box_size_um / self.pixel_size_y_um))
        
        # Storage for crop metadata
        self.crops: list[CropMetadata] = []
        
        # Napari components (initialized in run())
        self.viewer = None
        self.points_layer = None
        self.shapes_layer = None
        
        # Default contrast limits (will be computed from data)
        self.channel_contrasts: list[ChannelContrast] = []
        
        print(f"Crop box size: {box_size_um} µm = {self.box_width_px} x {self.box_height_px} pixels")
    
    def _compute_default_contrasts(self, data: np.ndarray):
        """Compute default contrast limits for each channel."""
        self.channel_contrasts = []
        n_channels = data.shape[0] if data.ndim >= 3 else 1
        
        if data.ndim == 2:
            self.channel_contrasts.append(ChannelContrast.from_data(data))
        else:
            for c in range(n_channels):
                if data.ndim == 3:  # CYX
                    channel_data = data[c]
                else:  # CZYX
                    channel_data = data[c]
                self.channel_contrasts.append(ChannelContrast.from_data(channel_data))
    
    def _get_box_corners(self, center_y: float, center_x: float) -> np.ndarray:
        """Get the four corners of a crop box given its center."""
        half_h = self.box_height_px / 2
        half_w = self.box_width_px / 2
        
        # Corners in clockwise order starting from top-left
        corners = np.array([
            [center_y - half_h, center_x - half_w],
            [center_y - half_h, center_x + half_w],
            [center_y + half_h, center_x + half_w],
            [center_y + half_h, center_x - half_w],
        ])
        return corners
    
    def _update_shapes(self, event=None):
        """Update the shapes layer to show boxes around all points."""
        if self.points_layer is None or self.shapes_layer is None:
            return
        
        points = self.points_layer.data
        if len(points) == 0:
            self.shapes_layer.data = []
            return
        
        # Create rectangles for each point
        rectangles = []
        for point in points:
            # Points are in (y, x) or (z, y, x) format depending on data dims
            if len(point) == 2:
                y, x = point
            else:
                y, x = point[-2], point[-1]  # Take last two coords (y, x)
            
            corners = self._get_box_corners(y, x)
            rectangles.append(corners)
        
        self.shapes_layer.data = rectangles
    
    def _on_points_change(self, event):
        """Callback when points are added/moved/removed."""
        self._update_shapes()
    
    def run(self):
        """
        Launch the napari viewer for interactive crop selection.
        
        Returns when the viewer is closed.
        """
        # Ensure napari is imported
        napari = _ensure_napari()
        
        # Prepare display data
        if self.z_project_for_display:
            display_data = self.czi_image.zmax()  # Shape: (C, Y, X)
        else:
            display_data = self.czi_image.image_data  # Shape: (C, Z, Y, X)
        
        # Compute default contrasts
        self._compute_default_contrasts(display_data)
        
        # Create viewer
        self.viewer = napari.Viewer(title=f"Crop Selector - {self.czi_image.name}")
        
        # Add image layers for each channel
        n_channels = display_data.shape[0]
        
        # Default colormaps for channels
        default_cmaps = ['green', 'magenta', 'cyan', 'yellow', 'red', 'blue']
        
        # Track image layers for later visibility check
        self.image_layers = []
        
        for c in range(n_channels):
            if display_data.ndim == 3:  # CYX
                channel_data = display_data[c]
            else:  # CZYX
                channel_data = display_data[c]
            
            cmap = default_cmaps[c % len(default_cmaps)]
            contrast = self.channel_contrasts[c]
            
            # Try to get channel name from wavelength info
            name = f"Channel {c}"
            if c < len(self.czi_image.channel_wavelengths):
                wl = self.czi_image.channel_wavelengths[c]
                if wl.get('emission'):
                    name = f"Ch{c} ({int(wl['emission'])}nm)"
            
            layer = self.viewer.add_image(
                channel_data,
                name=name,
                colormap=cmap,
                contrast_limits=(contrast.min_val, contrast.max_val),
                blending='additive',
            )
            self.image_layers.append(layer)
        
        # Add points layer for cell centers
        self.points_layer = self.viewer.add_points(
            ndim=2,
            name='Cell Centers',
            size=20,
            face_color='white',
            border_color='yellow',
            border_width=0.1,  # Relative to point size (napari default)
        )
        
        # Add shapes layer for crop box preview
        self.shapes_layer = self.viewer.add_shapes(
            ndim=2,
            name='Crop Boxes',
            face_color='transparent',
            edge_color='yellow',
            edge_width=2,
        )
        
        # Connect callback to update boxes when points change
        self.points_layer.events.data.connect(self._on_points_change)
        
        # Instructions
        print("\n" + "="*60)
        print("CROP SELECTION INSTRUCTIONS")
        print("="*60)
        print("• Click on cell centers to add crop points")
        print("• Yellow boxes show the crop region")
        print("• Select and drag points to adjust position")
        print("• Press Delete/Backspace to remove selected points")
        print("• Adjust channel contrast in the layer controls (left panel)")
        print("• HIDE channels you don't want in the final figure (👁 icon)")
        print("  - Hidden channels will be excluded from output")
        print("• Close the viewer when done selecting")
        print("="*60 + "\n")
        
        # Run napari (blocks until closed)
        napari.run()
        
        # After closing, collect the crops
        self._collect_crops()
    
    def _collect_crops(self):
        """Collect crop metadata from the current selection."""
        if self.points_layer is None:
            return
        
        points = self.points_layer.data
        self.crops = []
        
        # Get current contrast limits and visibility from image layers
        current_contrasts = []
        visible_channels = []
        
        for i, layer in enumerate(self.image_layers):
            limits = layer.contrast_limits
            current_contrasts.append(ChannelContrast(
                min_val=float(limits[0]),
                max_val=float(limits[1])
            ))
            if layer.visible:
                visible_channels.append(i)
        
        # Report which channels are included
        n_total = len(self.image_layers)
        n_visible = len(visible_channels)
        if n_visible < n_total:
            hidden = [i for i in range(n_total) if i not in visible_channels]
            print(f"Channels included: {visible_channels} (hidden: {hidden})")
        else:
            print(f"All {n_total} channels included")
        
        for i, point in enumerate(points):
            if len(point) == 2:
                center_y_px, center_x_px = int(point[0]), int(point[1])
            else:
                center_y_px, center_x_px = int(point[-2]), int(point[-1])
            
            # Convert to physical coordinates
            center_y_um = center_y_px * self.pixel_size_y_um
            center_x_um = center_x_px * self.pixel_size_x_um
            
            crop = CropMetadata(
                source_file=str(self.czi_image.name),
                source_name=self.czi_image.name,
                center_pixel=(center_y_px, center_x_px),
                box_size_pixels=(self.box_height_px, self.box_width_px),
                center_physical_um=(center_y_um, center_x_um),
                box_size_physical_um=self.box_size_um,
                pixel_size_um=(self.pixel_size_y_um, self.pixel_size_x_um),
                z_range=(0, self.czi_image.Z),
                z_projection="max",
                channel_contrasts=current_contrasts.copy(),
                channel_wavelengths=self.czi_image.channel_wavelengths.copy(),
                channel_colors_original=self.czi_image.channel_colors.copy(),
                visible_channels=visible_channels.copy(),
                crop_id=f"{self.czi_image.name}_crop{i:03d}",
            )
            self.crops.append(crop)
        
        print(f"Collected {len(self.crops)} crops")
    
    def save_session(self, output_path: str, visualization_config: Optional[dict] = None):
        """
        Save the crop session to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            visualization_config: Optional dict of display settings to save 
                                  (e.g., {'scale_bar_um': 10, 'border_pixels': 20})
        """
        session_data = {
            'source_file': str(self.czi_image.name),
            'box_size_um': self.box_size_um,
            'pixel_size_um': {
                'x': self.pixel_size_x_um,
                'y': self.pixel_size_y_um,
            },
            'timestamp': datetime.now().isoformat(),
            'crops': [crop.to_dict() for crop in self.crops],
            # Save the visualization config if provided, otherwise empty dict
            'visualization_config': visualization_config or {}
        }
        
        with open(output_path, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {output_path}")
    
    @classmethod
    def load_session(cls, session_path: str) -> list[CropMetadata]:
        """
        Load crops from a saved session file.
        
        Args:
            session_path: Path to the JSON session file
            
        Returns:
            List of CropMetadata objects
        """
        with open(session_path, 'r') as f:
            session_data = json.load(f)
        
        crops = [CropMetadata.from_dict(c) for c in session_data['crops']]
        print(f"Loaded {len(crops)} crops from {session_path}")
        return crops


def extract_crop(image_data: np.ndarray, crop_meta: CropMetadata) -> np.ndarray:
    """
    Extract a crop from image data based on metadata.
    
    Args:
        image_data: Full image array in CZYX format
        crop_meta: CropMetadata specifying the crop location
        
    Returns:
        Cropped array in CZYX format
    """
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels
    
    # Calculate bounds with boundary checking
    y_start = max(0, center_y - box_h // 2)
    y_end = min(image_data.shape[-2], center_y + box_h // 2)
    x_start = max(0, center_x - box_w // 2)
    x_end = min(image_data.shape[-1], center_x + box_w // 2)
    
    # Extract crop
    if image_data.ndim == 4:  # CZYX
        crop = image_data[:, :, y_start:y_end, x_start:x_end]
    elif image_data.ndim == 3:  # CYX (already projected)
        crop = image_data[:, y_start:y_end, x_start:x_end]
    else:
        raise ValueError(f"Unexpected image dimensions: {image_data.ndim}")
    
    return crop