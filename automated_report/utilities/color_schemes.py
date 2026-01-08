"""
Color scheme management for fluorescence microscopy figures.

Provides predefined color schemes and tools for creating custom ones.
Separates color rendering from raw data for easy figure regeneration.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json


@dataclass
class ChannelLUT:
    """
    Lookup table definition for a single channel.
    
    Can be defined by:
    - A named matplotlib colormap
    - An RGB color (for simple linear ramps from black to color)
    - A custom colormap object
    """
    name: str  # Display name
    color_rgb: Optional[tuple[float, float, float]] = None  # RGB 0-1
    colormap_name: Optional[str] = None  # matplotlib colormap name
    custom_cmap: Optional[LinearSegmentedColormap] = None  # Custom colormap
    
    def get_colormap(self) -> LinearSegmentedColormap:
        """Get the matplotlib colormap for this LUT."""
        if self.custom_cmap is not None:
            return self.custom_cmap
        
        if self.colormap_name is not None:
            return plt.get_cmap(self.colormap_name)
        
        if self.color_rgb is not None:
            # Create a linear ramp from black to the specified color
            r, g, b = self.color_rgb
            colors = [(0, 0, 0), (r, g, b)]
            return LinearSegmentedColormap.from_list(self.name, colors, N=256)
        
        # Default to grayscale
        return plt.get_cmap('gray')
    
    def apply(self, data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        Apply this LUT to normalized data.
        
        Args:
            data: 2D array of pixel values
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            
        Returns:
            RGBA array (H, W, 4) with values 0-1
        """
        # Normalize data
        if vmax > vmin:
            normalized = (data.astype(np.float32) - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(data, dtype=np.float32)
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 1)
        
        # Apply colormap
        cmap = self.get_colormap()
        return cmap(normalized)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary (custom cmaps not supported)."""
        return {
            'name': self.name,
            'color_rgb': self.color_rgb,
            'colormap_name': self.colormap_name,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ChannelLUT':
        """Deserialize from dictionary."""
        return cls(
            name=d['name'],
            color_rgb=tuple(d['color_rgb']) if d.get('color_rgb') else None,
            colormap_name=d.get('colormap_name'),
        )


@dataclass
class ColorScheme:
    """
    A complete color scheme for multi-channel images.
    
    Defines how each channel should be colored and how they combine
    in the merged image.
    """
    name: str
    channel_luts: list[ChannelLUT] = field(default_factory=list)
    merge_mode: str = "additive"  # "additive" or "maximum"
    background_color: tuple[float, float, float] = (0, 0, 0)  # RGB 0-1
    
    def get_lut(self, channel_index: int) -> ChannelLUT:
        """Get LUT for a specific channel, with fallback to grayscale."""
        if channel_index < len(self.channel_luts):
            return self.channel_luts[channel_index]
        # Fallback
        return ChannelLUT(name=f"Channel {channel_index}", colormap_name='gray')
    
    def render_channel(
        self,
        data: np.ndarray,
        channel_index: int,
        vmin: float,
        vmax: float
    ) -> np.ndarray:
        """
        Render a single channel with its LUT.
        
        Args:
            data: 2D array (Y, X) of a single channel
            channel_index: Which channel this is
            vmin, vmax: Contrast limits
            
        Returns:
            RGB array (H, W, 3) with values 0-1
        """
        lut = self.get_lut(channel_index)
        rgba = lut.apply(data, vmin, vmax)
        return rgba[:, :, :3]  # Drop alpha
    
    def render_merged(
        self,
        channel_data: list[np.ndarray],
        contrasts: list[tuple[float, float]]
    ) -> np.ndarray:
        """
        Render a merged multi-channel image.
        
        Args:
            channel_data: List of 2D arrays, one per channel
            contrasts: List of (vmin, vmax) tuples for each channel
            
        Returns:
            RGB array (H, W, 3) with values 0-255 as uint8
        """
        if len(channel_data) == 0:
            raise ValueError("No channel data provided")
        
        h, w = channel_data[0].shape
        merged = np.zeros((h, w, 3), dtype=np.float32)
        
        for i, (data, (vmin, vmax)) in enumerate(zip(channel_data, contrasts)):
            rgb = self.render_channel(data, i, vmin, vmax)
            
            if self.merge_mode == "additive":
                merged += rgb
            elif self.merge_mode == "maximum":
                merged = np.maximum(merged, rgb)
        
        # Clip and convert to uint8
        merged = np.clip(merged, 0, 1)
        return (merged * 255).astype(np.uint8)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'channel_luts': [lut.to_dict() for lut in self.channel_luts],
            'merge_mode': self.merge_mode,
            'background_color': self.background_color,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> 'ColorScheme':
        """Deserialize from dictionary."""
        return cls(
            name=d['name'],
            channel_luts=[ChannelLUT.from_dict(lut) for lut in d['channel_luts']],
            merge_mode=d.get('merge_mode', 'additive'),
            background_color=tuple(d.get('background_color', (0, 0, 0))),
        )


# =============================================================================
# PREDEFINED COLOR SCHEMES
# =============================================================================

def scheme_standard_rgb() -> ColorScheme:
    """Standard RGB scheme: Ch0=Red, Ch1=Green, Ch2=Blue."""
    return ColorScheme(
        name="Standard RGB",
        channel_luts=[
            ChannelLUT(name="Red", color_rgb=(1, 0, 0)),
            ChannelLUT(name="Green", color_rgb=(0, 1, 0)),
            ChannelLUT(name="Blue", color_rgb=(0, 0, 1)),
        ]
    )


def scheme_green_magenta() -> ColorScheme:
    """Colorblind-friendly: Ch0=Green, Ch1=Magenta."""
    return ColorScheme(
        name="Green-Magenta",
        channel_luts=[
            ChannelLUT(name="Green", color_rgb=(0, 1, 0)),
            ChannelLUT(name="Magenta", color_rgb=(1, 0, 1)),
        ]
    )


def scheme_cyan_magenta() -> ColorScheme:
    """Colorblind-friendly: Ch0=Cyan, Ch1=Magenta."""
    return ColorScheme(
        name="Cyan-Magenta",
        channel_luts=[
            ChannelLUT(name="Cyan", color_rgb=(0, 1, 1)),
            ChannelLUT(name="Magenta", color_rgb=(1, 0, 1)),
        ]
    )


def scheme_cyan_yellow() -> ColorScheme:
    """High contrast: Ch0=Cyan, Ch1=Yellow."""
    return ColorScheme(
        name="Cyan-Yellow",
        channel_luts=[
            ChannelLUT(name="Cyan", color_rgb=(0, 1, 1)),
            ChannelLUT(name="Yellow", color_rgb=(1, 1, 0)),
        ]
    )


def scheme_grayscale() -> ColorScheme:
    """All channels in grayscale (for individual display)."""
    return ColorScheme(
        name="Grayscale",
        channel_luts=[
            ChannelLUT(name="Gray", colormap_name='gray'),
            ChannelLUT(name="Gray", colormap_name='gray'),
            ChannelLUT(name="Gray", colormap_name='gray'),
            ChannelLUT(name="Gray", colormap_name='gray'),
        ]
    )


def scheme_publication_3ch() -> ColorScheme:
    """
    Publication-ready 3-channel scheme.
    Ch0=Blue (DAPI/Hoechst), Ch1=Green (GFP/Alexa488), Ch2=Red (RFP/Alexa594)
    """
    return ColorScheme(
        name="Publication 3-Channel",
        channel_luts=[
            ChannelLUT(name="DAPI", color_rgb=(0.2, 0.4, 1.0)),  # Softer blue
            ChannelLUT(name="GFP", color_rgb=(0, 1, 0)),
            ChannelLUT(name="RFP", color_rgb=(1, 0.2, 0.2)),  # Softer red
        ]
    )


def scheme_fire() -> ColorScheme:
    """Fire LUT for single-channel intensity display."""
    # Create a fire-like colormap
    colors = [
        (0, 0, 0),      # Black
        (0.5, 0, 0.5),  # Purple
        (1, 0, 0),      # Red
        (1, 0.5, 0),    # Orange
        (1, 1, 0),      # Yellow
        (1, 1, 1),      # White
    ]
    fire_cmap = LinearSegmentedColormap.from_list('fire', colors, N=256)
    
    return ColorScheme(
        name="Fire",
        channel_luts=[
            ChannelLUT(name="Fire", custom_cmap=fire_cmap),
        ]
    )


def scheme_custom(channel_colors: list[tuple[float, float, float]], name: str = "Custom") -> ColorScheme:
    """
    Create a custom color scheme from RGB tuples.
    
    Args:
        channel_colors: List of (R, G, B) tuples, values 0-1
        name: Name for the scheme
        
    Returns:
        ColorScheme with the specified colors
    """
    # Convert color names or hex strings to RGB tuples if needed
    def parse_color(color):
        if isinstance(color, str):
            # Try to parse common color names or hex codes
            import matplotlib.colors as mcolors
            try:
                return mcolors.to_rgb(color)
            except ValueError:
                raise ValueError(f"Invalid color string: {color}")
        elif isinstance(color, tuple) and len(color) == 3:
            return color
        else:
            raise ValueError(f"Invalid color format: {color}")

    parsed_colors = [parse_color(c) for c in channel_colors]

    luts = [
        ChannelLUT(name=f"Channel {i}", color_rgb=color)
        for i, color in enumerate(parsed_colors)
    ]
    return ColorScheme(name=name, channel_luts=luts)


def scheme_from_wavelengths(
    wavelengths: list[dict],
    color_map: Optional[dict[float, tuple[float, float, float]]] = None,
) -> ColorScheme:
    """
    Create a color scheme based on emission wavelengths.
    
    Args:
        wavelengths: List of wavelength dicts with 'emission' key (from CziImage.channel_wavelengths)
        color_map: Optional custom mapping of emission wavelength -> RGB color.
                   If not provided, uses sensible defaults.
    
    Returns:
        ColorScheme with colors assigned by wavelength
    
    Example:
        scheme = scheme_from_wavelengths(czi.channel_wavelengths)
    """
    # Default wavelength to color mapping
    # Maps emission wavelength ranges to display colors
    default_color_map = {
        # Blue/violet range (400-480nm) -> blue or gray (DAPI, Hoechst)
        405: (0.3, 0.3, 1.0),   # DAPI - blue
        444: (0.5, 0.5, 0.5),   # Hoechst - gray (often excluded from figures)
        450: (0.3, 0.3, 1.0),   # Blue
        461: (0.3, 0.3, 1.0),   # DAPI emission
        
        # Green range (500-550nm) -> green (GFP, Alexa488, FITC)
        509: (0, 1, 0),         # GFP
        519: (0, 1, 0),         # Alexa488
        524: (0, 1, 0),         # ~525nm -> green
        525: (0, 1, 0),         # Generic green
        530: (0, 1, 0),         # FITC
        
        # Yellow/orange range (550-600nm) -> magenta or red (RFP, Alexa555, Alexa568)
        560: (1, 0, 1),         # mCherry
        570: (1, 0, 1),         # Alexa555
        580: (1, 0, 1),         # 
        596: (1, 0, 1),         # ~596nm -> magenta
        597: (1, 0, 1),         # Alexa568
        
        # Red range (600-700nm) -> red or far-red
        620: (1, 0.2, 0.2),     # Texas Red
        668: (1, 0, 0.5),       # Alexa647
        700: (0.8, 0, 0.2),     # Far red
    }
    
    if color_map:
        default_color_map.update(color_map)
    
    def get_color_for_wavelength(emission_nm: float) -> tuple[float, float, float]:
        """Find the closest matching color for a given emission wavelength."""
        if emission_nm is None:
            return (0.5, 0.5, 0.5)  # Gray for unknown
        
        # Find closest wavelength in map
        closest_wl = min(default_color_map.keys(), key=lambda x: abs(x - emission_nm))
        
        # Only use if reasonably close (within 30nm)
        if abs(closest_wl - emission_nm) <= 30:
            return default_color_map[closest_wl]
        
        # Fall back to wavelength-based heuristic
        if emission_nm < 480:
            return (0.3, 0.3, 1.0)  # Blue
        elif emission_nm < 550:
            return (0, 1, 0)        # Green
        elif emission_nm < 620:
            return (1, 0, 1)        # Magenta
        else:
            return (1, 0.2, 0.2)    # Red
    
    luts = []
    for i, wl_info in enumerate(wavelengths):
        emission = wl_info.get('emission')
        color = get_color_for_wavelength(emission)
        name = f"{int(emission)}nm" if emission else f"Ch{i}"
        luts.append(ChannelLUT(name=name, color_rgb=color))
    
    return ColorScheme(name="Wavelength-based", channel_luts=luts)


# Registry of built-in schemes
BUILTIN_SCHEMES = {
    'rgb': scheme_standard_rgb,
    'green_magenta': scheme_green_magenta,
    'cyan_magenta': scheme_cyan_magenta,
    'cyan_yellow': scheme_cyan_yellow,
    'grayscale': scheme_grayscale,
    'publication_3ch': scheme_publication_3ch,
    'fire': scheme_fire,
}


def get_scheme(name: str) -> ColorScheme:
    """Get a built-in color scheme by name."""
    if name not in BUILTIN_SCHEMES:
        available = ', '.join(BUILTIN_SCHEMES.keys())
        raise ValueError(f"Unknown scheme '{name}'. Available: {available}")
    return BUILTIN_SCHEMES[name]()


def list_schemes() -> list[str]:
    """List all available built-in scheme names."""
    return list(BUILTIN_SCHEMES.keys())


def save_scheme(scheme: ColorScheme, path: str):
    """Save a color scheme to JSON file."""
    with open(path, 'w') as f:
        json.dump(scheme.to_dict(), f, indent=2)


def load_scheme(path: str) -> ColorScheme:
    """Load a color scheme from JSON file."""
    with open(path, 'r') as f:
        return ColorScheme.from_dict(json.load(f))