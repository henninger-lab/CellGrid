import glob
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import skimage.io
import tifffile


SUPPORTED_IMAGE_GLOBS = (
    "*.czi",
    "*.CZI",
    "*.ome.tif",
    "*.ome.tiff",
    "*.OME.TIF",
    "*.OME.TIFF",
)


def is_czi_path(file_path: os.PathLike[str] | str) -> bool:
    return str(file_path).lower().endswith(".czi")


def is_ome_tiff_path(file_path: os.PathLike[str] | str) -> bool:
    suffixes = str(file_path).lower()
    return suffixes.endswith(".ome.tif") or suffixes.endswith(".ome.tiff")


def microscopy_image_stem(file_path: os.PathLike[str] | str) -> str:
    name = Path(file_path).name
    lower_name = name.lower()
    for suffix in (".ome.tiff", ".ome.tif"):
        if lower_name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def find_supported_image_files(input_dir: str) -> list[str]:
    files: list[str] = []
    for pattern in SUPPORTED_IMAGE_GLOBS:
        files.extend(glob.glob(os.path.join(input_dir, pattern)))
    return sorted(set(files))


def get_fluorophore_name(excitation: float = None, emission: float = None) -> str:
    """
    Impute fluorophore name from excitation/emission wavelengths.
    """
    wl = emission if emission not in (None, 0) else excitation

    if wl is None or wl <= 0:
        return "Unknown"

    if wl < 500:
        return "Hoechst"
    elif wl < 570:
        return "mStayGold"
    elif wl < 640:
        return "mScarlet"
    else:
        return "Far-red"


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "0", "0.0"):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if parsed <= 0 else parsed


def _convert_length_to_meters(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None

    normalized = (unit or "").strip().lower()
    factors = {
        "m": 1.0,
        "meter": 1.0,
        "meters": 1.0,
        "mm": 1e-3,
        "millimeter": 1e-3,
        "millimeters": 1e-3,
        "um": 1e-6,
        "\u00b5m": 1e-6,
        "\u03bcm": 1e-6,
        "micrometer": 1e-6,
        "micrometers": 1e-6,
        "nm": 1e-9,
        "nanometer": 1e-9,
        "nanometers": 1e-9,
    }
    factor = factors.get(normalized)
    if factor is None:
        return value if normalized == "" else None
    return value * factor


def _ome_color_int_to_hex(value: str | int | None) -> str | None:
    if value in (None, ""):
        return None
    try:
        rgba = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    rgb = rgba & 0x00FFFFFF
    return f"#{rgb:06X}"


def _transpose_tiff_to_ctzyx(arr: np.ndarray, axes: str) -> np.ndarray:
    desired_axes = "CTZYX"
    axes = axes.upper()

    squeeze_axes: list[int] = []
    for idx, (axis_name, size) in enumerate(zip(axes, arr.shape)):
        if axis_name in desired_axes:
            continue
        if size != 1:
            raise ValueError(
                f"Unsupported TIFF axes '{axes}' with non-singleton axis '{axis_name}' "
                f"of size {size}."
            )
        squeeze_axes.append(idx)

    if squeeze_axes:
        arr = np.squeeze(arr, axis=tuple(squeeze_axes))
        axes = "".join(axis for axis in axes if axis in desired_axes)

    if "Y" not in axes or "X" not in axes:
        raise ValueError(f"Expected YX axes in TIFF series, got '{axes}'.")

    current_axes = [axis for axis in desired_axes if axis in axes]
    permutation = [axes.index(axis) for axis in current_axes]
    arr = np.transpose(arr, permutation)

    for axis_name in desired_axes:
        if axis_name not in current_axes:
            insert_at = desired_axes.index(axis_name)
            arr = np.expand_dims(arr, axis=insert_at)
            current_axes.insert(insert_at, axis_name)

    return arr


def _normalized_ctzyx_shape(shape: tuple[int, ...], axes: str) -> tuple[int, int, int, int, int]:
    axes = axes.upper()
    axis_sizes: dict[str, int] = {}
    for axis_name, size in zip(axes, shape):
        if axis_name in "CTZYX":
            axis_sizes[axis_name] = int(size)
        elif int(size) != 1:
            raise ValueError(
                f"Unsupported TIFF axes '{axes}' with non-singleton axis '{axis_name}' "
                f"of size {size}."
            )
    if "Y" not in axis_sizes or "X" not in axis_sizes:
        raise ValueError(f"Expected YX axes in TIFF series, got '{axes}'.")
    return tuple(axis_sizes.get(axis_name, 1) for axis_name in "CTZYX")


class CziImage:
    def __init__(self, file_path):
        """
        Initializes the CziImage object, loading either a CZI or aligned OME-TIFF.
        Keeps the older CziImage API intact while adding OME-TIFF compatibility.
        """
        self.file_path = str(file_path)
        self.path = Path(file_path)
        self.name = microscopy_image_stem(file_path)

        print("\n--- Loading microscopy file ---")
        print(f"Name: {file_path}")

        self._image_data = None
        self._channel_data_cache = {}
        self._zmax_cache = {}
        self._shape = None
        self._ome_data_ctzyx = None

        if is_czi_path(file_path):
            self._backend = "czi"
            self._init_czi(file_path)
        elif is_ome_tiff_path(file_path):
            self._backend = "ome_tiff"
            self._init_ome_tiff(file_path)
        else:
            raise ValueError(
                f"Unsupported microscopy file: {file_path}. "
                "Expected .czi, .ome.tif, or .ome.tiff."
            )

        self.parse_metadata()

    def _init_czi(self, file_path):
        import aicspylibczi

        self.czi = aicspylibczi.CziFile(file_path)
        self.dims = self.czi.dims
        self.size = self.czi.size
        self._dims_shape = self.czi.get_dims_shape()
        self._dims_shape0 = self._dims_shape[0]

        self.axis_dict = {}
        self.channel_indexes = {}
        for i, dim_name in enumerate(self.dims):
            self.axis_dict[dim_name] = self.size[i]
            self.channel_indexes[dim_name] = i

        for dim_info in self._dims_shape:
            for key, (start, size) in dim_info.items():
                if key in ["C", "Z", "Y", "X", "T"]:
                    setattr(self, key, size)

        self.C = self._dims_shape0.get("C", (0, 1))[1]
        self.Z = self._dims_shape0.get("Z", (0, 1))[1]
        self.Y = self._dims_shape0.get("Y", (0, 1))[1]
        self.X = self._dims_shape0.get("X", (0, 1))[1]
        self.T = self._dims_shape0.get("T", (0, 1))[1]

        print(f"Image dimensions: {self.dims}")
        print(f"Image size: {self.size}")
        print(f"Axes: {self._dims_shape}")

        self.axes = "CZYX"
        self.metadata_root = self.czi.meta
        self.metadata_xml = ET.tostring(self.metadata_root, encoding="unicode")

    def _init_ome_tiff(self, file_path):
        with tifffile.TiffFile(file_path) as tif:
            series = tif.series[0]
            self._ome_series_axes = series.axes.upper()
            self._ome_series_shape = tuple(int(v) for v in series.shape)
            self.metadata_xml = tif.ome_metadata or ""

        self.dims = self._ome_series_axes
        self.size = self._ome_series_shape
        self.axes = "CZYX"
        self.metadata_root = ET.fromstring(self.metadata_xml) if self.metadata_xml else None

        ctzyx_shape = _normalized_ctzyx_shape(self._ome_series_shape, self._ome_series_axes)
        self.C, self.T, self.Z, self.Y, self.X = ctzyx_shape
        self._dims_shape = [{
            "C": (0, self.C),
            "T": (0, self.T),
            "Z": (0, self.Z),
            "Y": (0, self.Y),
            "X": (0, self.X),
        }]
        self._dims_shape0 = self._dims_shape[0]
        self.axis_dict = {axis: size for axis, (_, size) in self._dims_shape0.items()}
        self.channel_indexes = {axis: idx for idx, axis in enumerate("CTZYX")}

        print(f"Image dimensions: {self.dims}")
        print(f"Image size: {self.size}")
        print(f"Normalized axes: {self._dims_shape}")

    @property
    def image_data(self):
        """
        Lazy load the full image data only when needed.
        Returns data in CZYX format using the first timepoint if T > 1.
        """
        if self._image_data is None:
            if self._backend == "czi":
                channel_stack = [
                    self.read_channel(channel, time_index=0)
                    for channel in range(self.C)
                ]
                self._image_data = np.stack(channel_stack, axis=0)
            else:
                self._image_data = self._load_ome_data_ctzyx()[:, 0]

            self.shape = self._image_data.shape

        return self._image_data

    @property
    def shape(self):
        if self._image_data is not None:
            return self._image_data.shape
        return (self.C, self.Z, self.Y, self.X)

    @shape.setter
    def shape(self, value):
        self._shape = value

    def _normalize_channels(self, channels):
        n_channels = self._dims_shape0.get("C", (0, 1))[1]

        if channels is None:
            return list(range(n_channels))

        if isinstance(channels, int):
            channels = [channels]
        else:
            channels = list(channels)

        if not channels:
            raise ValueError("At least one channel must be requested.")

        invalid = [channel for channel in channels if channel < 0 or channel >= n_channels]
        if invalid:
            raise ValueError(
                f"Requested invalid channel(s) {invalid}; available channel range is 0-{n_channels - 1}."
            )

        return channels

    def _build_read_kwargs(self, time_index=0):
        read_kwargs = {}

        for dim in ["B", "V", "S", "M", "H", "I", "R"]:
            if dim in self._dims_shape0:
                read_kwargs[dim] = 0

        if "T" in self._dims_shape0:
            t_size = self._dims_shape0["T"][1]
            if t_size > 1:
                print(f"  Timelapse detected ({t_size} frames), using frame {time_index}")
            read_kwargs["T"] = min(time_index, t_size - 1)

        return read_kwargs

    def _load_ome_data_ctzyx(self):
        if self._ome_data_ctzyx is None:
            with tifffile.TiffFile(self.file_path) as tif:
                series = tif.series[0]
                arr = series.asarray()
            self._ome_data_ctzyx = _transpose_tiff_to_ctzyx(arr, self._ome_series_axes)
        return self._ome_data_ctzyx

    def read_channel(self, channel, time_index=0):
        cache_key = (channel, time_index)
        if cache_key in self._channel_data_cache:
            return self._channel_data_cache[cache_key]

        if self._backend == "czi":
            n_z = self._dims_shape0.get("Z", (0, 1))[1]
            read_kwargs = self._build_read_kwargs(time_index)
            if "C" in self._dims_shape0:
                read_kwargs["C"] = channel
            channel_image, _ = self.czi.read_image(**read_kwargs)
            squeezed = np.squeeze(channel_image)

            if squeezed.ndim == 2:
                channel_data = squeezed[np.newaxis, :, :]
            elif squeezed.ndim == 3 and n_z > 1:
                channel_data = squeezed
            else:
                while squeezed.ndim > 3:
                    squeezed = np.squeeze(squeezed, axis=0)
                if squeezed.ndim == 2:
                    channel_data = squeezed[np.newaxis, :, :]
                elif squeezed.ndim == 3:
                    channel_data = squeezed
                else:
                    raise ValueError(f"Unexpected channel shape for C={channel}: {squeezed.shape}")
        else:
            data_ctzyx = self._load_ome_data_ctzyx()
            t_index = min(time_index, data_ctzyx.shape[1] - 1)
            channel_data = data_ctzyx[channel, t_index]

        self._channel_data_cache[cache_key] = channel_data
        return channel_data

    def zmax_channel(self, channel, time_index=0):
        cache_key = (channel, time_index)
        if cache_key in self._zmax_cache:
            return self._zmax_cache[cache_key]

        channel_data = self.read_channel(channel, time_index=time_index)
        if channel_data.shape[0] > 1:
            print(f"  Computing max projection over {channel_data.shape[0]} Z-slices")
            zmax = np.max(channel_data, axis=0)
        else:
            zmax = channel_data[0]

        self._zmax_cache[cache_key] = zmax
        return zmax

    def zmax(self, time_index=0, channels=None):
        channels = self._normalize_channels(channels)
        max_proj = np.stack(
            [self.zmax_channel(channel, time_index=time_index) for channel in channels],
            axis=0,
        )
        print(f"  Max projection shape: {max_proj.shape} (CYX) from channels {channels}")
        return max_proj

    def get_channel_slice(self, channel):
        return self.read_channel(channel)

    def getChannelSlice(self, channel):
        return self.get_channel_slice(channel)

    def _parse_czi_metadata(self, verbose=False):
        root = self.metadata_root

        self.channel_colors = []
        self.channel_wavelengths = []
        self.channel_names = []

        channel_nodes = root.findall(".//Information/Image/Dimensions/Channels/Channel")
        for node in channel_nodes:
            try:
                self.channel_names.append(node.get("Name") or f"Channel {len(self.channel_names)}")

                color_node = node.find("Color")
                if color_node is not None:
                    self.channel_colors.append(color_node.text)
                else:
                    self.channel_colors.append(None)

                excitation_node = node.find("ExcitationWavelength")
                emission_node = node.find("EmissionWavelength")
                wavelengths = {
                    "excitation": _safe_float(excitation_node.text if excitation_node is not None else None),
                    "emission": _safe_float(emission_node.text if emission_node is not None else None),
                }
                self.channel_wavelengths.append(wavelengths)
            except Exception as e:
                if verbose:
                    print(f"Couldn't find specific colors or wavelengths in the metadata: {e}")

        self.scale = {}
        scaling_items = root.find(".//Scaling/Items")
        if scaling_items is not None:
            for distance in scaling_items.findall("Distance"):
                axis_id = distance.get("Id")
                value_node = distance.find("Value")
                scale_value = _safe_float(value_node.text if value_node is not None else None)
                if axis_id and scale_value is not None:
                    self.scale[axis_id] = scale_value

    def _parse_ome_metadata(self):
        self.channel_colors = []
        self.channel_wavelengths = []
        self.channel_names = []
        self.scale = {}

        if self.metadata_root is None:
            return

        namespace_uri = self.metadata_root.tag.split("}")[0].strip("{")
        ns = {"ome": namespace_uri} if namespace_uri else {}
        prefix = "ome:" if ns else ""

        pixels = self.metadata_root.find(f".//{prefix}Pixels", ns)
        if pixels is not None:
            for axis_id in ("X", "Y", "Z"):
                value = _safe_float(pixels.get(f"PhysicalSize{axis_id}"))
                unit = pixels.get(f"PhysicalSize{axis_id}Unit")
                meters = _convert_length_to_meters(value, unit)
                if meters is not None:
                    self.scale[axis_id] = meters

            for idx, channel_node in enumerate(pixels.findall(f"{prefix}Channel", ns)):
                self.channel_names.append(channel_node.get("Name") or f"Channel {idx}")
                self.channel_colors.append(_ome_color_int_to_hex(channel_node.get("Color")))
                self.channel_wavelengths.append({
                    "excitation": _safe_float(channel_node.get("ExcitationWavelength")),
                    "emission": _safe_float(channel_node.get("EmissionWavelength")),
                })

        while len(self.channel_names) < self.C:
            channel_index = len(self.channel_names)
            self.channel_names.append(f"Channel {channel_index}")
            self.channel_colors.append(None)
            self.channel_wavelengths.append({"excitation": None, "emission": None})

    def parse_metadata(self, verbose=False):
        if self._backend == "czi":
            self._parse_czi_metadata(verbose=verbose)
        else:
            self._parse_ome_metadata()

        self.channel_fluorophores = []
        for wl in self.channel_wavelengths:
            fluor = get_fluorophore_name(wl.get("excitation"), wl.get("emission"))
            self.channel_fluorophores.append(fluor)

        print(f"Channel fluorophores: {self.channel_fluorophores}")
        print("--- Color data ---")
        print(f"Channel wavelengths: {self.channel_wavelengths}")
        print(f"Channel colors: {self.channel_colors}")
        print("-----------------------\n")

        print("--- Pixel sizes ---")
        if "X" in self.scale:
            print(f"X Scale (microns per pixel): {self.scale['X'] * 1e6:.4f}")
        if "Y" in self.scale:
            print(f"Y Scale (microns per pixel): {self.scale['Y'] * 1e6:.4f}")
        if "Z" in self.scale:
            print(f"Z Scale (microns per pixel): {self.scale['Z'] * 1e6:.4f}")
        print("-----------------------\n")

    def save_numpy(self):
        output_file_path = f"{self.name}.npy"
        try:
            np.save(output_file_path, self.image_data)
            print(f"Image data successfully saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving image data to file: {e}")

    def save_metadata_xml(self, output_file_path):
        try:
            with open(output_file_path, "w", encoding="utf-8") as f:
                f.write(self.metadata_xml)
            print(f"XML metadata successfully saved to {output_file_path}")
        except IOError as e:
            print(f"Error saving XML metadata to file: {e}")

    def save_as_rgb_tiff(self, output_directory):
        max_projected_array = self.zmax()
        rgb_image = np.zeros((max_projected_array.shape[1], max_projected_array.shape[2], 3), dtype=np.uint16)

        channel_order = {"R": 0, "G": 1, "B": 2}
        for color, index in channel_order.items():
            if index >= max_projected_array.shape[0]:
                continue
            plane_index = {"R": 0, "G": 1, "B": 2}[color]
            channel_data = max_projected_array[index, :, :]

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

    def check_actual_bit_depth(self):
        image_data = self.image_data
        unique_values = len(np.unique(image_data))
        max_value = image_data.max()

        print(f"Unique values in image: {unique_values}")
        print(f"Maximum value: {max_value}")
        print(f"Minimum value: {image_data.min()}")

        if max_value <= 255:
            print("Data appears to be 8-bit")
        elif max_value <= 4095:
            print("Data appears to be 12-bit (stored in 16-bit container)")
        elif max_value <= 16383:
            print("Data appears to be 14-bit (stored in 16-bit container)")
        elif max_value <= 65535:
            print("Data uses 16-bit range")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(image_data.flatten(), bins=256, log=True)
        plt.xlabel("Pixel Value")
        plt.ylabel("Count (log scale)")
        plt.title("Histogram of Pixel Values")
        plt.axvline(x=4095, color="r", linestyle="--", label="12-bit max")
        plt.axvline(x=16383, color="g", linestyle="--", label="14-bit max")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python parse_czi.py <path_to_microscopy_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        my_czi_image = CziImage(file_path)
        print(f"Successfully read the image file! Axes: {my_czi_image.axes}")
        output_xml_path = f"{file_path}.xml"
        my_czi_image.save_metadata_xml(output_xml_path)

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
