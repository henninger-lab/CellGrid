import aicspylibczi
import numpy as np
import xml.etree.ElementTree as ET
import sys
import skimage.io
import os
from pathlib import Path


class CziImage:
    def __init__(self, file_path):
        """
        Initializes the CziImage object, loading the image and metadata.
        Uses aicspylibczi which has native support for zstd-compressed CZI files.
        """
        print("\n--- Loading CZI file ---")
        print(f"Name: {file_path}")
        
        # Use aicspylibczi which supports zstd compression
        self.czi = aicspylibczi.CziFile(file_path)
        
        # Get dimensions
        self.dims = self.czi.dims  # e.g., 'STCZYX' or 'BSCZYX'
        self.size = self.czi.size
        dims_shape = self.czi.get_dims_shape()
        
        path = Path(file_path)
        self.name = path.stem
        
        # Build axis dict from dims_shape
        self.axis_dict = {}
        self.channel_indexes = {}
        for i, dim_name in enumerate(self.dims):
            self.axis_dict[dim_name] = self.size[i]
            self.channel_indexes[dim_name] = i
        
        # Extract CZYX dimensions
        for dim_info in dims_shape:
            for key, (start, size) in dim_info.items():
                if key in ['C', 'Z', 'Y', 'X']:
                    setattr(self, key, size)
        
        print(f"Image dimensions: {self.dims}")
        print(f"Image size: {self.size}")
        print(f"Axes: {dims_shape}")
        
        # Store the actual axes as CZYX for compatibility
        self.axes = 'CZYX'
        
        # Load metadata
        # aicspylibczi returns an ElementTree.Element, not a string
        self.metadata_root = self.czi.meta
        
        # Convert to string for compatibility with save_metadata_xml
        self.metadata_xml = ET.tostring(self.metadata_root, encoding='unicode')
        
        self.parse_metadata()
        
        # Cache for full image data
        self._image_data = None
    
    @property
    def image_data(self):
        """
        Lazy load the full image data only when needed.
        Returns data in CZYX format.
        """
        if self._image_data is None:
            # Read the entire image
            full_image, _ = self.czi.read_image()
            
            # Squeeze out dimensions of size 1, but keep C, Z, Y, X
            # The returned shape might be like (B, S, C, Z, Y, X)
            # We want to reduce it to (C, Z, Y, X)
            squeezed = np.squeeze(full_image)
            
            # Ensure we have 4D array in CZYX order
            if squeezed.ndim == 2:
                # Single channel, single Z-slice: YX -> CZYX
                squeezed = squeezed[np.newaxis, np.newaxis, :, :]
            elif squeezed.ndim == 3:
                # Could be CYX or ZYX, assume CYX
                squeezed = squeezed[:, np.newaxis, :, :]
            
            self._image_data = squeezed
            self.shape = self._image_data.shape
            
        return self._image_data
    
    @property
    def shape(self):
        """Get the shape of the image data."""
        if self._image_data is not None:
            return self._image_data.shape
        # Return expected shape without loading data
        return (self.C, self.Z, self.Y, self.X)
    
    @shape.setter
    def shape(self, value):
        """Allow setting shape."""
        self._shape = value
    
    def zmax(self, time_index=0):
        """
        Returns max projection along Z axis.
        For timelapse data, uses the specified time index (default: first frame).
        Returns data in CYX format.
        """
        full_image, _ = self.czi.read_image()
        
        # Build axis mapping for the full image
        axis_map = {dim: i for i, dim in enumerate(self.dims)}
        current_shape = list(full_image.shape)
        
        # Handle Time dimension - select single timepoint
        if 'T' in axis_map:
            t_axis = axis_map['T']
            t_size = current_shape[t_axis]
            if t_size > 1:
                print(f"  Timelapse detected ({t_size} frames), using frame {time_index}")
            full_image = np.take(full_image, min(time_index, t_size - 1), axis=t_axis)
            # Remove T from axis_map and shift indices
            del axis_map['T']
            axis_map = {k: (v if v < t_axis else v - 1) for k, v in axis_map.items()}
        
        # Handle Scene/Block/Mosaic and other extra dimensions - take first
        for dim in ['S', 'B', 'M', 'H', 'V', 'I', 'R']:
            if dim in axis_map:
                dim_axis = axis_map[dim]
                dim_size = full_image.shape[dim_axis]
                if dim_size > 1:
                    print(f"  Multiple {dim} detected ({dim_size}), using first")
                full_image = np.take(full_image, 0, axis=dim_axis)
                del axis_map[dim]
                axis_map = {k: (v if v < dim_axis else v - 1) for k, v in axis_map.items()}
        
        # Now we should have CZYX or subset
        # Find Z axis and do max projection
        if 'Z' in axis_map:
            z_axis = axis_map['Z']
            z_size = full_image.shape[z_axis]
            if z_size > 1:
                print(f"  Computing max projection over {z_size} Z-slices")
                max_proj = np.max(full_image, axis=z_axis)
            else:
                max_proj = np.take(full_image, 0, axis=z_axis)
            del axis_map['Z']
            axis_map = {k: (v if v < z_axis else v - 1) for k, v in axis_map.items()}
        else:
            max_proj = full_image
        
        # Now we should have CYX (or YX if no C)
        # Ensure C is first axis
        if 'C' in axis_map:
            c_axis = axis_map['C']
            if c_axis != 0:
                max_proj = np.moveaxis(max_proj, c_axis, 0)
        else:
            # No C dimension, add it
            if max_proj.ndim == 2:
                max_proj = max_proj[np.newaxis, :, :]
        
        # Final shape check - should be (C, Y, X)
        if max_proj.ndim != 3:
            print(f"  Warning: unexpected shape {max_proj.shape}, attempting to fix...")
            # Try to squeeze any remaining singleton dimensions
            max_proj = np.squeeze(max_proj)
            if max_proj.ndim == 2:
                max_proj = max_proj[np.newaxis, :, :]
            elif max_proj.ndim > 3:
                # Take first element of extra dimensions
                while max_proj.ndim > 3:
                    max_proj = max_proj[0]
        
        print(f"  Max projection shape: {max_proj.shape} (CYX)")
        return max_proj
    
    def get_channel_slice(self, channel):
        """Get a specific channel from the image data."""
        return self.image_data[channel, :, :, :]
    
    def parse_metadata(self, verbose=False):
        """
        Parses the XML metadata to extract useful information.
        """
        # Use the already-parsed root element
        root = self.metadata_root
        
        # --- Extract Channel Names and Colors ---
        self.channel_colors = []
        self.channel_wavelengths = []
        
        # This XPath looks for Channel nodes within the Information/Image/Dimensions path
        channel_nodes = root.findall('.//Information/Image/Dimensions/Channels/Channel')
        for node in channel_nodes:
            try:
                color_node = node.find('Color')
                if color_node is not None:
                    self.channel_colors.append(color_node.text)
                else:
                    self.channel_colors.append(None)
                
                excitation_node = node.find('ExcitationWavelength')
                emission_node = node.find('EmissionWavelength')
                wavelengths = {
                    'excitation': float(excitation_node.text) if excitation_node is not None else None,
                    'emission': float(emission_node.text) if emission_node is not None else None
                }
                self.channel_wavelengths.append(wavelengths)
            except Exception as e:
                if verbose:
                    print(f"Couldn't find specific colors or wavelengths in the metadata: {e}")
        
        # --- Extract Pixel Scaling Information ---
        self.scale = {}
        # This XPath looks for the Scaling/Items path
        scaling_items = root.find('.//Scaling/Items')
        if scaling_items is not None:
            for distance in scaling_items.findall('Distance'):
                axis_id = distance.get('Id')
                scale_value = float(distance.find('Value').text)
                self.scale[axis_id] = scale_value
        
        print("--- Color data ---")
        print(f'Channel wavelengths: {self.channel_wavelengths}')
        print(f"Channel colors: {self.channel_colors}")
        print("-----------------------\n")
        
        print("--- Pixel sizes ---")
        
        # Scaling info
        if 'X' in self.scale:
            print(f"X Scale (microns per pixel): {self.scale['X'] * 1e6:.4f}")
        if 'Y' in self.scale:
            print(f"Y Scale (microns per pixel): {self.scale['Y'] * 1e6:.4f}")
        if 'Z' in self.scale:
            print(f"Z Scale (microns per pixel): {self.scale['Z'] * 1e6:.4f}")
        
        print("-----------------------\n")
    
    def save_numpy(self):
        '''Saves the image data as a numpy array file. Axis order is CZYX.'''
        output_file_path = f"{self.name}.npy"
        try:
            np.save(output_file_path, self.image_data)
            print(f"Image data successfully saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving image data to file: {e}")
    
    def save_metadata_xml(self, output_file_path):
        try:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(self.metadata_xml)
            print(f"XML metadata successfully saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving XML metadata to file: {e}")
    
    def save_as_rgb_tiff(self, output_directory):
        """
        Creates a max-projected RGB image and saves it as a single TIFF.
        
        Args:
            output_directory (str): The directory to save the file in.
        """
        # Create the max projection first
        max_projected_array = self.zmax()  # Shape (C, Y, X)
        
        # Create an empty RGB image array of shape (Y, X, 3)
        rgb_image = np.zeros((max_projected_array.shape[1], max_projected_array.shape[2], 3), dtype=np.uint16)
        
        # Assign channels to the RGB planes
        channel_order = {'R': 0, 'G': 1, 'B': 2}
        for color, index in channel_order.items():
            if index >= max_projected_array.shape[0]:
                continue
            plane_index = {'R': 0, 'G': 1, 'B': 2}[color]
            channel_data = max_projected_array[index, :, :]
            
            # Normalize and scale this specific channel to uint16
            min_val, max_val = channel_data.min(), channel_data.max()
            if max_val > min_val:
                normalized = (channel_data - min_val) / (max_val - min_val)
            else:
                normalized = channel_data.astype(np.float32)
            
            rgb_image[:, :, plane_index] = (normalized * 65535).astype(np.uint16)
        
        base_name = os.path.splitext(os.path.basename(self.name))[0]
        output_tiff_path = os.path.join(output_directory, f"{base_name}_RGB_composite.tif")
        print(f"Saving RGB composite TIFF to: {output_tiff_path}")
        skimage.io.imsave(output_tiff_path, rgb_image, check_contrast=False)
    
    def check_actual_bit_depth(self):  # For debugging only
        """
        Check the actual bit depth usage of the data.
        Many microscopes save 12-bit or 14-bit data in 16-bit containers.
        """
        image_data = self.image_data
        unique_values = len(np.unique(image_data))
        max_value = image_data.max()
        
        print(f"Unique values in image: {unique_values}")
        print(f"Maximum value: {max_value}")
        print(f"Minimum value: {image_data.min()}")
        
        # Check common bit depths
        if max_value <= 255:
            print("Data appears to be 8-bit")
        elif max_value <= 4095:
            print("Data appears to be 12-bit (stored in 16-bit container)")
        elif max_value <= 16383:
            print("Data appears to be 14-bit (stored in 16-bit container)")
        elif max_value <= 65535:
            print("Data uses 16-bit range")
        
        # Show histogram of values
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(image_data.flatten(), bins=256, log=True)
        plt.xlabel('Pixel Value')
        plt.ylabel('Count (log scale)')
        plt.title('Histogram of Pixel Values')
        plt.axvline(x=4095, color='r', linestyle='--', label='12-bit max')
        plt.axvline(x=16383, color='g', linestyle='--', label='14-bit max')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_czi.py <path_to_czi_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        # Create an instance of our new CziImage class
        my_czi_image = CziImage(file_path)
        
        # Now you can access the attributes of your object
        print(f"Successfully read the CZI file! Axes: {my_czi_image.axes}")
        output_xml_path = f"{file_path}.xml"  # Example: save XML next to CZI file with .xml extension
        my_czi_image.save_metadata_xml(output_xml_path)
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()