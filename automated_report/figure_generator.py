"""
Figure generation for representative microscopy images.

Creates publication-ready figures with:
- Side-by-side individual channels + merged
- Scale bars
- Multi-row grids combining multiple experiments
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.figure import Figure
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import json

from .crop_selector import CropMetadata, ChannelContrast, extract_crop
from .color_schemes import ColorScheme, get_scheme


@dataclass
class ScaleBarConfig:
    """Configuration for scale bars."""
    length_um: float = 10.0  # Length in microns
    height_fraction: float = 0.02  # Height as fraction of image height
    color: str = 'white'
    position: str = 'bottom_right'  # 'bottom_right', 'bottom_left', 'top_right', 'top_left'
    margin_fraction: float = 0.05  # Margin from edge as fraction of image size
    show_label: bool = True
    label_color: str = 'white'
    font_size: int = 10


@dataclass 
class PanelConfig:
    """Configuration for a figure panel."""
    show_channel_labels: bool = False  # Default off - cleaner figures
    channel_label_color: str = 'white'
    channel_label_position: str = 'top_left'  # 'top_left', 'top_right'
    channel_label_fontsize: int = 12
    merged_label: str = "Merge"
    gap_pixels: int = 10  # Gap between sub-panels
    gap_color: tuple[int, int, int] = (255, 255, 255)  # White gaps
    border_pixels: int = 10  # Border around the entire panel
    scale_bar: Optional[ScaleBarConfig] = None


def add_scale_bar(
    ax: plt.Axes,
    pixel_size_um: float,
    image_shape: tuple[int, int],
    config: ScaleBarConfig,
    show_label: Optional[bool] = None,  # Override config.show_label if specified
):
    """
    Add a scale bar to a matplotlib axes.
    
    Args:
        ax: Matplotlib axes
        pixel_size_um: Size of one pixel in microns
        image_shape: (height, width) of the image in pixels
        config: Scale bar configuration
        show_label: Override whether to show label (None = use config.show_label)
    """
    h, w = image_shape
    bar_length_px = config.length_um / pixel_size_um
    bar_height_px = h * config.height_fraction
    
    margin_x = w * config.margin_fraction
    margin_y = h * config.margin_fraction
    
    # Calculate position
    if 'right' in config.position:
        x = w - margin_x - bar_length_px
    else:
        x = margin_x
    
    if 'bottom' in config.position:
        y = h - margin_y - bar_height_px
    else:
        y = margin_y
    
    # Draw the bar
    rect = Rectangle(
        (x, y), bar_length_px, bar_height_px,
        facecolor=config.color, edgecolor='none'
    )
    ax.add_patch(rect)
    
    # Add label (use override if provided, otherwise use config)
    should_show_label = show_label if show_label is not None else config.show_label
    if should_show_label:
        label_x = x + bar_length_px / 2
        if 'bottom' in config.position:
            label_y = y - bar_height_px * 0.5
            va = 'bottom'
        else:
            label_y = y + bar_height_px * 1.5
            va = 'top'
        
        ax.text(
            label_x, label_y, f"{config.length_um:.0f} µm",
            color=config.label_color,
            fontsize=config.font_size,
            ha='center', va=va,
            fontweight='bold'
        )


def add_channel_label(
    ax: plt.Axes,
    label: str,
    image_shape: tuple[int, int],
    config: PanelConfig
):
    """Add a channel label to the corner of an image."""
    h, w = image_shape
    margin = 0.02
    
    if 'left' in config.channel_label_position:
        x = w * margin
        ha = 'left'
    else:
        x = w * (1 - margin)
        ha = 'right'
    
    if 'top' in config.channel_label_position:
        y = h * margin
        va = 'top'
    else:
        y = h * (1 - margin)
        va = 'bottom'
    
    ax.text(
        x, y, label,
        color=config.channel_label_color,
        fontsize=config.channel_label_fontsize,
        ha=ha, va=va,
        fontweight='bold'
    )


class FigurePanel:
    """
    A single figure panel showing individual channels side-by-side plus merged.
    
    Layout: [Ch0] [Ch1] [Ch2] ... [Merged]
    
    Only channels marked as visible in crop_meta.visible_channels are included.
    """
    
    def __init__(
        self,
        crop_data: np.ndarray,
        crop_meta: CropMetadata,
        color_scheme: ColorScheme,
        config: Optional[PanelConfig] = None,
        channel_names: Optional[list[str]] = None,
    ):
        """
        Initialize a figure panel.
        
        Args:
            crop_data: Image data in CYX format (already max-projected if needed)
            crop_meta: Metadata for this crop
            color_scheme: Color scheme to use for rendering
            config: Panel configuration (uses defaults if None)
            channel_names: Optional list of channel names for labels
        """
        self.crop_data = crop_data
        self.crop_meta = crop_meta
        self.color_scheme = color_scheme
        self.config = config or PanelConfig()
        
        # Get active (visible) channels
        self.active_channels = crop_meta.get_active_channels()
        self.n_channels = len(self.active_channels)
        
        # Always print channel info for debugging
        total_channels = crop_data.shape[0]
        print(f"  [FigurePanel] visible_channels={crop_meta.visible_channels}, "
              f"active={self.active_channels}, showing {self.n_channels}/{total_channels}")
        
        # Set up channel names (only for active channels)
        if channel_names is not None:
            self.channel_names = [channel_names[i] for i in self.active_channels if i < len(channel_names)]
        else:
            # Try to get from wavelengths or use defaults
            self.channel_names = []
            for idx, orig_c in enumerate(self.active_channels):
                if orig_c < len(crop_meta.channel_wavelengths):
                    wl = crop_meta.channel_wavelengths[orig_c]
                    if wl.get('emission'):
                        self.channel_names.append(f"{int(wl['emission'])}nm")
                    else:
                        self.channel_names.append(f"Ch{orig_c}")
                else:
                    self.channel_names.append(f"Ch{orig_c}")

    def should_render_merged(self) -> bool:
        """Return True when a merged panel would add information."""
        return len(self.active_channels) > 1
    
    def render_individual_channels(self) -> list[np.ndarray]:
        """
        Render each visible channel individually.
        
        Returns:
            List of RGB arrays (H, W, 3) as uint8
        """
        rendered = []
        
        for panel_idx, orig_c in enumerate(self.active_channels):
            channel_data = self.crop_data[orig_c]
            contrast = self.crop_meta.channel_contrasts[orig_c]
            
            # Use panel_idx for color scheme (so colors are consistent regardless of which channels are hidden)
            rgb = self.color_scheme.render_channel(
                channel_data, panel_idx, contrast.min_val, contrast.max_val
            )
            rendered.append((rgb * 255).astype(np.uint8))
        
        return rendered
    
    def render_merged(self) -> np.ndarray:
        """
        Render the merged multi-channel image (visible channels only).
        
        Returns:
            RGB array (H, W, 3) as uint8
        """
        channel_list = [self.crop_data[orig_c] for orig_c in self.active_channels]
        contrasts = [
            (self.crop_meta.channel_contrasts[orig_c].min_val,
             self.crop_meta.channel_contrasts[orig_c].max_val)
            for orig_c in self.active_channels
        ]
        
        return self.color_scheme.render_merged(channel_list, contrasts)
    
    def render_composite(self, include_merged: bool = True) -> np.ndarray:
        """
        Render the complete side-by-side composite image.
        
        Args:
            include_merged: Whether to include the merged image at the end
            
        Returns:
            RGB array of the complete panel as uint8
        """
        include_merged = include_merged and self.should_render_merged()
        individual = self.render_individual_channels()
        if include_merged:
            merged = self.render_merged()
            all_images = individual + [merged]
        else:
            all_images = individual
        
        # Get dimensions
        h, w = all_images[0].shape[:2]
        n_images = len(all_images)
        gap = self.config.gap_pixels
        border = self.config.border_pixels
        
        # Calculate composite size
        total_w = n_images * w + (n_images - 1) * gap + 2 * border
        total_h = h + 2 * border
        
        # Create composite
        composite = np.zeros((total_h, total_w, 3), dtype=np.uint8)
        composite[:, :] = self.config.gap_color
        
        # Place images
        for i, img in enumerate(all_images):
            x_start = border + i * (w + gap)
            composite[border:border+h, x_start:x_start+w] = img
        
        return composite
    
    def _draw_scale_bar_on_array(
        self, 
        img: np.ndarray, 
        x_offset: int, 
        y_offset: int,
        panel_width: int,
        panel_height: int,
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Draw a scale bar directly onto a numpy array (no matplotlib).
        
        Args:
            img: RGB array to draw on (modified in place)
            x_offset: X position of the panel within img
            y_offset: Y position of the panel within img
            panel_width: Width of the panel in pixels
            panel_height: Height of the panel in pixels
            show_label: Whether to draw the label text
            
        Returns:
            Modified image array
        """
        config = self.config.scale_bar or ScaleBarConfig()
        pixel_size_um = self.crop_meta.pixel_size_um[1]  # X pixel size
        
        bar_length_px = int(config.length_um / pixel_size_um)
        bar_height_px = max(3, int(panel_height * config.height_fraction))
        
        margin_x = int(panel_width * config.margin_fraction)
        margin_y = int(panel_height * config.margin_fraction)
        
        # Calculate position within the panel
        if 'right' in config.position:
            bar_x = panel_width - margin_x - bar_length_px
        else:
            bar_x = margin_x
        
        if 'bottom' in config.position:
            bar_y = panel_height - margin_y - bar_height_px
        else:
            bar_y = margin_y
        
        # Convert to global coordinates
        global_x = x_offset + bar_x
        global_y = y_offset + bar_y
        
        # Draw the bar (white rectangle)
        img[global_y:global_y+bar_height_px, global_x:global_x+bar_length_px] = (255, 255, 255)
        
        return img
    
    def save_full_resolution(
        self,
        output_path: str,
        include_merged: bool = True,
        add_scale_bars: bool = True,
    ):
        """
        Save the panel at FULL RESOLUTION without any resampling.
        
        This bypasses matplotlib entirely and saves the raw pixel data.
        Scale bars are drawn directly onto the image array.
        
        NOTE: This method does NOT include text labels on scale bars.
        For labels, use save() with PDF format and add text in Illustrator,
        or use save() with PNG for a complete figure with text.
        
        Args:
            output_path: Output file path (.png or .tiff recommended)
            include_merged: Include merged image
            add_scale_bars: Draw scale bars on the image
        """
        import skimage.io
        
        # Get full-resolution composite
        include_merged = include_merged and self.should_render_merged()
        composite = self.render_composite(include_merged=include_merged)
        
        if add_scale_bars:
            # Get panel dimensions
            individual = self.render_individual_channels()
            h, w = individual[0].shape[:2]
            n_panels = len(individual) + (1 if include_merged else 0)
            gap = self.config.gap_pixels
            border = self.config.border_pixels
            
            # Draw scale bar on each panel
            for i in range(n_panels):
                x_offset = border + i * (w + gap)
                y_offset = border
                
                # Only show label on first panel (but we can't draw text easily without PIL/matplotlib)
                # So we just draw the bar on all panels
                self._draw_scale_bar_on_array(
                    composite, x_offset, y_offset, w, h, show_label=False
                )
        
        # Save directly
        skimage.io.imsave(output_path, composite, check_contrast=False)
        print(f"Saved full-resolution panel to {output_path} ({composite.shape[1]}x{composite.shape[0]} pixels)")
    
    def save_individual_images(
        self,
        output_dir: str,
        prefix: str = "",
        include_merged: bool = True,
        add_scale_bars: bool = True,
    ):
        """
        Save each channel and merged image as separate full-resolution files.
        
        This is ideal for Illustrator workflows where you want to:
        - Place each image separately
        - Have full control over layout
        - Apply different effects to individual channels
        
        Args:
            output_dir: Directory to save images
            prefix: Prefix for filenames (e.g., "cell01_")
            include_merged: Include the merged image
            add_scale_bars: Draw scale bars on images
            
        Returns:
            List of saved file paths
        """
        import skimage.io
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Render all images
        individual = self.render_individual_channels()
        
        h, w = individual[0].shape[:2]
        
        # Save individual channels
        for i, (img, orig_c) in enumerate(zip(individual, self.active_channels)):
            # Add scale bar if requested
            if add_scale_bars:
                img = img.copy()
                self._draw_scale_bar_on_array(img, 0, 0, w, h, show_label=False)
            
            # Get channel name for filename
            if orig_c < len(self.crop_meta.channel_wavelengths):
                wl = self.crop_meta.channel_wavelengths[orig_c]
                if wl.get('emission'):
                    ch_name = f"{int(wl['emission'])}nm"
                else:
                    ch_name = f"ch{orig_c}"
            else:
                ch_name = f"ch{orig_c}"
            
            filename = output_path / f"{prefix}{ch_name}.png"
            skimage.io.imsave(str(filename), img, check_contrast=False)
            saved_files.append(str(filename))
        
        # Save merged
        include_merged = include_merged and self.should_render_merged()
        if include_merged:
            merged = self.render_merged()
            if add_scale_bars:
                merged = merged.copy()
                self._draw_scale_bar_on_array(merged, 0, 0, w, h, show_label=False)
            
            filename = output_path / f"{prefix}merged.png"
            skimage.io.imsave(str(filename), merged, check_contrast=False)
            saved_files.append(str(filename))
        
        print(f"Saved {len(saved_files)} images to {output_dir}/ ({w}x{h} pixels each)")
        return saved_files

    def create_figure(
        self,
        include_merged: bool = True,
        dpi: int = 150,
        add_scale_bar_to: str = 'all_label_first',  # 'all_label_first', 'all', 'merged', 'first', 'none'
    ) -> Figure:
        """
        Create a matplotlib figure with the panel.
        
        Args:
            include_merged: Include merged image
            dpi: Figure DPI
            add_scale_bar_to: Where to add scale bars:
                - 'all_label_first': Scale bar on all panels, label only on first (default)
                - 'all': Scale bar with label on all panels
                - 'merged': Only on merged image
                - 'first': Only on first channel
                - 'none': No scale bars
            
        Returns:
            Matplotlib Figure object
        """
        include_merged = include_merged and self.should_render_merged()
        individual = self.render_individual_channels()
        merged = self.render_merged() if include_merged else None
        
        n_cols = self.n_channels + (1 if include_merged else 0)
        h, w = individual[0].shape[:2]
        
        # Calculate figure size
        panel_width_in = w / dpi
        panel_height_in = h / dpi
        fig_width = panel_width_in * n_cols
        fig_height = panel_height_in
        
        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(fig_width, fig_height),
            dpi=dpi
        )
        
        if n_cols == 1:
            axes = [axes]
        
        # Configure scale bar
        scale_bar_config = self.config.scale_bar or ScaleBarConfig()
        pixel_size_um = self.crop_meta.pixel_size_um[1]  # Use X pixel size
        
        # Plot individual channels
        for i, (img, ax) in enumerate(zip(individual, axes[:self.n_channels])):
            ax.imshow(img)
            ax.axis('off')
            
            if self.config.show_channel_labels:
                add_channel_label(ax, self.channel_names[i], (h, w), self.config)
            
            # Determine if we should add scale bar and label
            if add_scale_bar_to in ['all', 'all_label_first']:
                show_label = (i == 0) if add_scale_bar_to == 'all_label_first' else None
                add_scale_bar(ax, pixel_size_um, (h, w), scale_bar_config, show_label=show_label)
            elif add_scale_bar_to == 'first' and i == 0:
                add_scale_bar(ax, pixel_size_um, (h, w), scale_bar_config)
        
        # Plot merged
        if include_merged:
            ax = axes[-1]
            ax.imshow(merged)
            ax.axis('off')
            
            if self.config.show_channel_labels:
                add_channel_label(ax, self.config.merged_label, (h, w), self.config)
            
            if add_scale_bar_to == 'merged':
                add_scale_bar(ax, pixel_size_um, (h, w), scale_bar_config)
            elif add_scale_bar_to in ['all', 'all_label_first']:
                # For merged, don't show label if 'all_label_first'
                show_label = False if add_scale_bar_to == 'all_label_first' else None
                add_scale_bar(ax, pixel_size_um, (h, w), scale_bar_config, show_label=show_label)
        
        plt.tight_layout(pad=0)
        return fig
    
    def save(
        self,
        output_path: str,
        include_merged: bool = True,
        dpi: int = 300,
        add_scale_bar_to: str = 'all_label_first',
    ):
        """
        Save the panel to a file.
        
        Supports PNG, PDF, SVG, and other formats based on file extension.
        PDF and SVG outputs are vector-based and editable in Adobe Illustrator.
        
        Args:
            output_path: Output file path (extension determines format)
            include_merged: Include merged image
            dpi: DPI for raster elements
            add_scale_bar_to: Where to add scale bars
        """
        fig = self.create_figure(
            include_merged=include_merged,
            dpi=dpi,
            add_scale_bar_to=add_scale_bar_to,
        )
        
        # Determine format from extension
        ext = output_path.lower().split('.')[-1]
        
        if ext == 'pdf':
            # PDF with editable text for Illustrator
            fig.savefig(
                output_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0.02,
                backend='pdf',
            )
        elif ext == 'svg':
            # SVG is fully vector and editable
            fig.savefig(
                output_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0.02,
            )
        else:
            # PNG and other raster formats
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        
        plt.close(fig)
        print(f"Saved panel to {output_path}")


class FigureGrid:
    """
    Assembles multiple panels into a grid figure.
    
    Typically used to show multiple cells/conditions in rows,
    with individual channels + merged in columns.
    """
    
    def __init__(
        self,
        row_labels: Optional[list[str]] = None,
        column_gap: int = 5,
        row_gap: int = 5,
        label_fontsize: int = 14,
        show_row_labels: bool = True,
    ):
        """
        Initialize the figure grid.
        
        Args:
            row_labels: Labels for each row
            column_gap: Gap between columns in pixels
            row_gap: Gap between rows in pixels
            label_fontsize: Font size for row labels
            show_row_labels: Whether to show row labels
        """
        self.row_labels = row_labels or []
        self.column_gap = column_gap
        self.row_gap = row_gap
        self.label_fontsize = label_fontsize
        self.show_row_labels = show_row_labels
        
        # Storage for rows
        self.rows: list[FigurePanel] = []
    
    def add_row(self, panel: FigurePanel, label: Optional[str] = None):
        """Add a panel as a new row."""
        self.rows.append(panel)
        if label is not None:
            if len(self.row_labels) < len(self.rows):
                self.row_labels.append(label)
            else:
                self.row_labels[len(self.rows) - 1] = label
    
    def render_composite(self, include_merged: bool = True) -> np.ndarray:
        """
        Render the complete grid as a single image.
        
        Returns:
            RGB array as uint8
        """
        if len(self.rows) == 0:
            raise ValueError("No rows added to grid")
        
        # Render all row panels
        row_images = [
            panel.render_composite(include_merged=include_merged)
            for panel in self.rows
        ]
        
        # Get dimensions
        max_w = max(img.shape[1] for img in row_images)
        heights = [img.shape[0] for img in row_images]
        total_h = sum(heights) + self.row_gap * (len(heights) - 1)
        
        # Create composite
        composite = np.zeros((total_h, max_w, 3), dtype=np.uint8)
        
        y = 0
        for img in row_images:
            h = img.shape[0]
            w = img.shape[1]
            composite[y:y+h, :w] = img
            y += h + self.row_gap
        
        return composite
    
    def create_figure(
        self,
        include_merged: bool = True,
        dpi: int = 150,
        add_scale_bar_to: str = 'last_row',  # 'last_row', 'all', 'none'
    ) -> Figure:
        """
        Create a matplotlib figure with the grid.
        
        Args:
            include_merged: Include merged images
            dpi: Figure DPI
            add_scale_bar_to: Where to add scale bars
            
        Returns:
            Matplotlib Figure
        """
        n_rows = len(self.rows)
        if n_rows == 0:
            raise ValueError("No rows added to grid")

        include_merged = include_merged and self.rows[0].should_render_merged()
        
        # Determine number of columns from first row
        n_cols = self.rows[0].n_channels + (1 if include_merged else 0)
        
        # Get panel dimensions from first row
        h, w = self.rows[0].crop_data.shape[1:3]
        
        # Calculate figure size
        panel_width_in = w / dpi
        panel_height_in = h / dpi
        
        fig_width = panel_width_in * n_cols + (0.5 if self.show_row_labels else 0)
        fig_height = panel_height_in * n_rows
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(fig_width, fig_height),
            dpi=dpi
        )
        
        # Ensure axes is 2D array
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each row
        for row_idx, panel in enumerate(self.rows):
            individual = panel.render_individual_channels()
            merged = panel.render_merged() if include_merged else None
            
            scale_bar_config = panel.config.scale_bar or ScaleBarConfig()
            pixel_size_um = panel.crop_meta.pixel_size_um[1]
            
            is_last_row = row_idx == n_rows - 1
            
            # Plot individual channels
            for col_idx, img in enumerate(individual):
                ax = axes[row_idx, col_idx]
                ax.imshow(img)
                ax.axis('off')
                
                # Channel labels only on first row
                if row_idx == 0 and panel.config.show_channel_labels:
                    ax.set_title(
                        panel.channel_names[col_idx],
                        fontsize=panel.config.channel_label_fontsize,
                        color=panel.config.channel_label_color,
                        pad=2
                    )
                
                # Scale bar
                should_add_scale_bar = (
                    (add_scale_bar_to == 'all') or
                    (add_scale_bar_to == 'last_row' and is_last_row and col_idx == 0)
                )
                if should_add_scale_bar:
                    add_scale_bar(ax, pixel_size_um, (h, w), scale_bar_config)
            
            # Plot merged
            if include_merged:
                ax = axes[row_idx, -1]
                ax.imshow(merged)
                ax.axis('off')
                
                if row_idx == 0 and panel.config.show_channel_labels:
                    ax.set_title(
                        panel.config.merged_label,
                        fontsize=panel.config.channel_label_fontsize,
                        color=panel.config.channel_label_color,
                        pad=2
                    )
            
            # Row label
            if self.show_row_labels and row_idx < len(self.row_labels):
                axes[row_idx, 0].set_ylabel(
                    self.row_labels[row_idx],
                    fontsize=self.label_fontsize,
                    rotation=90,
                    va='center'
                )
        
        plt.tight_layout(pad=0.5)
        return fig
    
    def save(
        self,
        output_path: str,
        include_merged: bool = True,
        dpi: int = 300,
        add_scale_bar_to: str = 'last_row',
    ):
        """
        Save the grid to a file.
        
        Supports PNG, PDF, SVG, and other formats based on file extension.
        PDF and SVG outputs are vector-based and editable in Adobe Illustrator.
        
        Args:
            output_path: Output file path (extension determines format)
            include_merged: Include merged image
            dpi: DPI for raster elements
            add_scale_bar_to: Where to add scale bars
        """
        fig = self.create_figure(
            include_merged=include_merged,
            dpi=dpi,
            add_scale_bar_to=add_scale_bar_to,
        )
        
        # Determine format from extension
        ext = output_path.lower().split('.')[-1]
        
        if ext == 'pdf':
            # PDF with editable text for Illustrator
            fig.savefig(
                output_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0.05,
                backend='pdf',
            )
        elif ext == 'svg':
            # SVG is fully vector and editable
            fig.savefig(
                output_path, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0.05,
            )
        else:
            # PNG and other raster formats
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        
        plt.close(fig)
        print(f"Saved grid to {output_path}")


def generate_panel_from_crop(
    czi_image,
    crop_meta: CropMetadata,
    color_scheme: Optional[ColorScheme] = None,
    panel_config: Optional[PanelConfig] = None,
    channel_names: Optional[list[str]] = None,
    use_wavelength_colors: bool = True,
) -> FigurePanel:
    """
    Convenience function to generate a panel from a CziImage and crop metadata.
    
    Args:
        czi_image: CziImage instance
        crop_meta: Crop metadata (from selection session)
        color_scheme: Color scheme (if None, uses wavelength-based auto-assignment)
        panel_config: Panel configuration
        channel_names: Optional channel names
        use_wavelength_colors: If True and no color_scheme provided, assign colors
                               based on emission wavelengths (524nm->green, 596nm->magenta, etc.)
        
    Returns:
        FigurePanel ready for rendering
    """
    from .color_schemes import scheme_from_wavelengths
    
    # Extract crop from full image
    full_data = czi_image.image_data  # CZYX
    
    # Get crop bounds
    center_y, center_x = crop_meta.center_pixel
    box_h, box_w = crop_meta.box_size_pixels
    
    y_start = max(0, center_y - box_h // 2)
    y_end = min(full_data.shape[-2], center_y + box_h // 2)
    x_start = max(0, center_x - box_w // 2)
    x_end = min(full_data.shape[-1], center_x + box_w // 2)
    
    # Extract and max-project
    crop_czyx = full_data[:, :, y_start:y_end, x_start:x_end]
    crop_cyx = np.max(crop_czyx, axis=1)  # Max projection along Z
    
    # Determine color scheme
    if color_scheme is None:
        if use_wavelength_colors and crop_meta.channel_wavelengths:
            color_scheme = scheme_from_wavelengths(crop_meta.channel_wavelengths)
            print(f"  [Auto-color] Assigned colors based on wavelengths")
        else:
            color_scheme = get_scheme('green_magenta')
    
    # Default panel config with scale bar
    if panel_config is None:
        panel_config = PanelConfig(
            scale_bar=ScaleBarConfig(length_um=10.0)
        )
    
    return FigurePanel(
        crop_data=crop_cyx,
        crop_meta=crop_meta,
        color_scheme=color_scheme,
        config=panel_config,
        channel_names=channel_names,
    )
