#!/usr/bin/env python3
"""
Integrated Image Loader with Landsat Metadata Support and Resampling

This module provides functionality to load images from various file formats,
including a special branch for Landsat metadata files (.txt). When a Landsat
metadata file is selected, the script extracts the metadata (using the
LandsatMetadataReader class), searches for all band image files in the same
folder, loads each band (using PIL), and stacks them into a multiband image.
If bands differ in spatial dimensions, they are resampled to a common target
shape using bilinear interpolation.
"""

import os
import logging
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np

try:
    from osgeo import gdal
except ImportError:
    gdal = None

try:
    import spectral as spy
except ImportError:
    spy = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import glymur
except ImportError:
    glymur = None

try:
    import h5py
except ImportError:
    h5py = None

try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None

try:
    import exifread
except ImportError:
    exifread = None

from PyQt5.QtWidgets import QInputDialog

# Import resize for resampling bands.
from skimage.transform import resize



from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # Disable the limit



# Configure logging
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
def _parse_envi_header_fixed(file_path: str) -> Dict[str, Any]:
    """
    Parse an ENVI header file and return a dictionary of header parameters.
    Handles multi-line values and formatting quirks.
    """
    hdr: Dict[str, Any] = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()

    key: Optional[str] = None
    value_lines = []
    in_multiline = False

    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue

        if '=' in line and not in_multiline:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if value.startswith('{') and not value.endswith('}'):
                in_multiline = True
                value_lines = [value]
            else:
                hdr[key] = value.strip('{}').strip()
        elif in_multiline:
            value_lines.append(line)
            if '}' in line:
                in_multiline = False
                full_value = " ".join(value_lines)
                if key is not None:
                    hdr[key] = full_value.strip('{}').strip()
    return hdr


# -----------------------------------------------------------------------------
def _load_envi_raw(header_filepath: str, data_filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load ENVI raw data given header and data file paths.

    Returns:
        Tuple of (image data as a numpy array, header dictionary).
    """
    hdr = _parse_envi_header_fixed(header_filepath)
    try:
        samples = int(hdr.get('samples', 0))
        lines = int(hdr.get('lines', 0))
        bands = int(hdr.get('bands', 1))
    except Exception as e:
        raise ValueError(f"Failed to parse image dimensions from header: {e}")

    try:
        data_type_code = int(hdr.get('data type', 1))
    except Exception:
        data_type_code = 1

    data_type_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                     5: np.float64, 12: np.uint16, 13: np.uint32}
    dtype = data_type_map.get(data_type_code, np.uint8)
    header_offset = int(hdr.get('header offset', 0))
    interleave = hdr.get('interleave', 'bsq').lower()

    with open(data_filepath, 'rb') as f:
        f.seek(header_offset)
        data = np.fromfile(f, dtype=dtype)

    expected_size = samples * lines * bands
    if data.size < expected_size:
        raise IOError(f"Data size ({data.size}) is smaller than expected ({expected_size}).")
    data = data[:expected_size]

    if interleave == 'bsq':
        data = data.reshape((bands, lines, samples))
        data = np.transpose(data, (1, 2, 0))
    elif interleave == 'bil':
        data = data.reshape((lines, bands, samples))
        data = np.transpose(data, (0, 2, 1))
    elif interleave == 'bip':
        data = data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unknown interleave type: {interleave}")

    return data, hdr


# -----------------------------------------------------------------------------
class LandsatMetadataReader:
    """
    A class to read Landsat metadata from a .txt file and locate the band image files.
    """

    def __init__(self, metadata_file_path: str):
        """
        Initialize with the given Landsat metadata .txt file.
        """
        self.metadata_file_path = metadata_file_path
        self.metadata: Dict[str, Any] = {}
        self.bands: Dict[str, Dict[str, Any]] = {}

        # Predefined calibration values.
        solar_irradiances = {
            'LANDSAT_4': {'1': 1957, '2': 1825, '3': 1557, '4': 1033, '5': 214.9, '7': 80.72},
            'LANDSAT_5': {'1': 1957, '2': 1826, '3': 1554, '4': 1036, '5': 215.0, '7': 80.67},
            'LANDSAT_7': {'1': 1970, '2': 1842, '3': 1547, '4': 1044, '5': 225.7, '7': 82.06, '8': 1369},
            'LANDSAT_8': {'1': 1895.33, '2': 2004.57, '3': 1820.75, '4': 1549.49,
                          '5': 951.76, '6': 366.97, '7': 247.55, '8': 85.46, '9': 1723.88},
            'LANDSAT_9': {'1': 1895.33, '2': 2004.57, '3': 1820.75, '4': 1549.49,
                          '5': 951.76, '6': 366.97, '7': 247.55, '8': 85.46, '9': 1723.88}
        }

        wavelengths = {
            'LANDSAT_4': {'1': 0.485, '2': 0.56, '3': 0.66, '4': 0.83, '5': 1.65, '6': 11.45, '7': 2.215},
            'LANDSAT_5': {'1': 0.485, '2': 0.56, '3': 0.66, '4': 0.83, '5': 1.65, '6': 11.45, '7': 2.215},
            'LANDSAT_7': {'1': 0.4775, '2': 0.560, '3': 0.6615, '4': 0.835,
                          '5': 1.648, '6_VCID_1': 11.45, '6_VCID_2': 11.45, '7': 11.335, '8': 0.7055},
            'LANDSAT_8': {'1': 0.44, '2': 0.48, '3': 0.56, '4': 0.655, '5': 0.865,
                          '6': 1.61, '7': 2.20, '8': 0.59, '9': 1.37, '10': 10.9, '11': 12.05},
            'LANDSAT_9': {'1': 0.44, '2': 0.48, '3': 0.56, '4': 0.655, '5': 0.865,
                          '6': 1.61, '7': 2.20, '8': 0.59, '9': 1.37, '10': 10.9, '11': 12.05}
        }

        center_sensor_zenith = {
            'LANDSAT_4': 0.001,
            'LANDSAT_5': 0.001,
            'LANDSAT_7': 0.001,
            'LANDSAT_8': 0.001,
            'LANDSAT_9': 0.001,
        }

        # Read the metadata file and build a metadata dictionary.
        with open(self.metadata_file_path, 'r') as f:
            for line in f.readlines():
                if 'GROUP' in line or '=' not in line:
                    continue
                # Normalize key by removing spaces and converting to uppercase.
                line_normalized = line.replace(' ', '')
                items = line_normalized.split('=')
                key = items[0].strip().upper()
                value = items[1].replace('\n', '').replace('\"', '').strip()
                self.metadata[key] = value

        # Define band groups.
        all_bands: List[Union[int, str]] = []
        reflectance_bands: List[Union[int, str]] = []
        thermal_bands: List[Union[int, str]] = []

        spacecraft = self.metadata.get('SPACECRAFT_ID', '').upper()
        if spacecraft == 'LANDSAT_9':
            all_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            reflectance_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            thermal_bands = [10, 11]
        elif spacecraft == 'LANDSAT_8':
            all_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            reflectance_bands = [1, 2, 3, 4, 5, 6, 7, 8, 9]
            thermal_bands = [10, 11]
        elif spacecraft == 'LANDSAT_7':
            all_bands = [1, 2, 3, 4, 5, '6_VCID_1', '6_VCID_2', 7, 8]
            reflectance_bands = [1, 2, 3, 4, 5, 7, 8]
            thermal_bands = ['6_VCID_1', '6_VCID_2']
        elif spacecraft == 'LANDSAT_5':
            all_bands = [1, 2, 3, 4, 5, 6, 7]
            reflectance_bands = [1, 2, 3, 4, 5, 7]
            thermal_bands = [6]
        elif spacecraft == 'LANDSAT_4':
            all_bands = [1, 2, 3, 4, 5, 6, 7]
            reflectance_bands = [1, 2, 3, 4, 5, 7]
            thermal_bands = [6]
        else:
            raise KeyError(f"Invalid metadata file: unknown SPACECRAFT_ID '{spacecraft}'")

        # Populate the bands dictionary.
        for band in all_bands:
            band_str = str(band)
            self.bands[band_str] = {}
            self.bands[band_str]['file_name'] = self.metadata.get(f'FILE_NAME_BAND_{band}', '')
            self.bands[band_str]['number'] = band
            self.bands[band_str]['radiance_maximum'] = float(self.metadata.get(f'RADIANCE_MAXIMUM_BAND_{band}', 0))
            self.bands[band_str]['radiance_minimum'] = float(self.metadata.get(f'RADIANCE_MINIMUM_BAND_{band}', 0))
            self.bands[band_str]['quantize_cal_maximum'] = float(self.metadata.get(f'QUANTIZE_CAL_MAX_BAND_{band}', 0))
            self.bands[band_str]['quantize_cal_minimum'] = float(self.metadata.get(f'QUANTIZE_CAL_MIN_BAND_{band}', 0))
            self.bands[band_str]['radiance_mult'] = float(self.metadata.get(f'RADIANCE_MULT_BAND_{band}', 0))
            self.bands[band_str]['radiance_add'] = float(self.metadata.get(f'RADIANCE_ADD_BAND_{band}', 0))
            self.bands[band_str]['wavelength'] = wavelengths[spacecraft][band_str]
            self.bands[band_str]['center_sensor_zenith'] = center_sensor_zenith[spacecraft]

            if band in reflectance_bands:
                self.bands[band_str]['saturation'] = self.metadata.get(f'SATURATION_BAND_{band}', '')
                self.bands[band_str]['reflectance_maximum'] = float(
                    self.metadata.get(f'REFLECTANCE_MAXIMUM_BAND_{band}', 0))
                self.bands[band_str]['reflectance_minimum'] = float(
                    self.metadata.get(f'REFLECTANCE_MINIMUM_BAND_{band}', 0))
                self.bands[band_str]['reflectance_mult'] = float(self.metadata.get(f'REFLECTANCE_MULT_BAND_{band}', 0))
                self.bands[band_str]['reflectance_add'] = float(self.metadata.get(f'REFLECTANCE_ADD_BAND_{band}', 0))
                self.bands[band_str]['solar_irradiance'] = solar_irradiances[spacecraft][band_str]
                self.bands[band_str]['type'] = 'reflectance'

            if band in thermal_bands:
                self.bands[band_str]['k1_constant'] = float(self.metadata.get(f'K1_CONSTANT_BAND_{band}', 0))
                self.bands[band_str]['k2_constant'] = float(self.metadata.get(f'K2_CONSTANT_BAND_{band}', 0))
                self.bands[band_str]['type'] = 'thermal'

    def get_band_metadata_by_file_name(self, file_name: str) -> dict:
        """
        Retrieve the band metadata corresponding to the given file name.
        """
        for band in self.bands.values():
            if os.path.basename(file_name) == band.get('file_name', ''):
                return band
        return {}

    def find_band_files(self) -> dict:
        """
        Search for band image files in the same folder as the metadata file.
        Constructs full paths using the 'file_name' entries from the metadata.
        Returns:
            A dictionary mapping band numbers (as strings) to the full file paths.
        """
        band_files = {}
        folder = os.path.dirname(self.metadata_file_path)
        for band, data in self.bands.items():
            file_name = data.get('file_name', '')
            if file_name:
                full_path = os.path.join(folder, file_name)
                if os.path.exists(full_path):
                    band_files[band] = full_path
                else:
                    logging.warning(f"File for band {band} not found: {full_path}")
        return band_files


# -----------------------------------------------------------------------------
def resample_band(band: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resample a band image to the target shape using bilinear interpolation.

    Args:
        band (np.ndarray): Input band image.
        target_shape (tuple): Desired (height, width).

    Returns:
        np.ndarray: Resampled image.
    """
    resampled = resize(band, target_shape, order=1, preserve_range=True, anti_aliasing=True)
    return resampled.astype(band.dtype)


# -----------------------------------------------------------------------------
class ImageLoader:
    """
    Loads image data from various file formats, including support for Landsat
    metadata files that reference multiple band images.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath: str = filepath
        self.image_data: Optional[np.ndarray] = None
        self.num_bands: int = 1
        self.metadata: Dict[str, Any] = {}

    def load(self) -> np.ndarray:
        """
        Load image data using the appropriate method based on file extension.
        """
        ext = os.path.splitext(self.filepath)[1].lower()

        if ext == "":
            base = os.path.splitext(self.filepath)[0]
            hdr_candidate = base + ".hdr"
            if os.path.exists(hdr_candidate):
                logging.info("Detected raw data file with accompanying ENVI header.")
                result, hdr = _load_envi_raw(hdr_candidate, self.filepath)
                self.image_data = result
                self.metadata = hdr
                self.num_bands = 1 if self.image_data.ndim == 2 else self.image_data.shape[2]
                self._check_for_external_metadata()
                return self.image_data

        # Landsat metadata file branch.
        if ext == ".txt":
            return self._load_landsat_metadata()

        if ext in [".tif", ".bil"]:
            result = self._load_with_rasterio()
        elif ext == ".jp2":
            result = self._load_with_glymur()
        elif ext == ".img":
            result = self._load_with_gdal()
        elif ext in [".hdr", ".dat", ".envi"]:
            result = self._load_with_spectral()
        elif ext == ".raw":
            result = self._load_raw_custom()
        elif ext == ".hdf":
            result = self._load_with_h5py()
        elif ext == ".nc":
            result = self._load_with_netcdf4()
        elif ext == ".l1r":
            result = self._load_with_h5py_l1r()
        else:
            if Image is not None:
                try:
                    with Image.open(self.filepath) as img:
                        img_array = np.array(img)
                        if img_array.ndim == 2:
                            self.image_data = img_array
                            self.num_bands = 1
                        elif img_array.ndim == 3:
                            self.image_data = img_array
                            self.num_bands = img_array.shape[2]
                        result = self.image_data
                except Exception as e:
                    logging.error(f"PIL load failed: {e}")
                    raise IOError(f"Unsupported file format: {ext}")
            else:
                raise IOError(f"Unsupported file format: {ext}")

        self._check_for_external_metadata()
        return result

    def _load_landsat_metadata(self) -> np.ndarray:
        """
        Load a Landsat metadata file (.txt) by using the LandsatMetadataReader
        to extract metadata and locate the corresponding band files. Each band file
        is loaded (using PIL) and then stacked to form a multiband image.
        """
        if Image is None:
            raise ImportError("PIL is not available for loading Landsat band files.")

        # Use LandsatMetadataReader to parse the metadata file.
        reader = LandsatMetadataReader(self.filepath)
        band_files = reader.find_band_files()

        if not band_files:
            raise IOError("No band files found from Landsat metadata.")

        bands_list: List[np.ndarray] = []

        # Sort the bands by their numeric value first, then by name.
        def sort_key(b: str):
            try:
                # Numeric bands come first, sorted by integer value
                return (0, int(b))
            except ValueError:
                # Non-numeric (e.g. "6_VCID_1") come after, sorted lexicographically
                return (1, b)

        for band in sorted(band_files.keys(), key=sort_key):
            band_path = band_files[band]
            logging.info(f"Loading band {band} from {band_path}")
            with Image.open(band_path) as img:
                band_array = np.array(img)
                bands_list.append(band_array)

        if not bands_list:
            raise IOError("Failed to load any band files from Landsat metadata.")

        # Use the shape of the first band as the target shape.
        target_shape = bands_list[0].shape[:2]
        resampled_bands = []
        for idx, band in enumerate(bands_list):
            if band.shape[:2] != target_shape:
                logging.info(f"Resampling band {idx + 1} from shape {band.shape[:2]} to {target_shape}")
                band = resample_band(band, target_shape)
            resampled_bands.append(band)

        # Stack bands along the last axis.
        if resampled_bands[0].ndim == 2:
            stacked = np.stack(resampled_bands, axis=-1)
        else:
            stacked = resampled_bands[0]
        self.image_data = stacked
        self.metadata.update(reader.metadata)
        self.num_bands = stacked.shape[-1] if stacked.ndim == 3 else 1
        return self.image_data

    def _load_with_pil(self, path: str) -> np.ndarray:
        """
        Load an image from the specified path using PIL.
        """
        if Image is None:
            raise ImportError("PIL is not available.")
        with Image.open(path) as img:
            return np.array(img)

    def get_wavelengths(self) -> Optional[list]:
        """
        Extract wavelengths from metadata if available.
        """
        if self.metadata and 'wavelength' in self.metadata:
            raw = self.metadata['wavelength']
            if isinstance(raw, str):
                raw = raw.strip('{}')
                try:
                    wavelengths = [float(val.strip()) for val in raw.split(',')]
                    return wavelengths
                except Exception as e:
                    logging.error(f"Error parsing wavelengths: {e}")
                    return None
            elif isinstance(raw, list):
                try:
                    return [float(val) for val in raw]
                except Exception as e:
                    logging.error(f"Error converting wavelength list: {e}")
                    return None
        return None

    def _load_with_rasterio(self) -> np.ndarray:
        if rasterio is None:
            raise ImportError("rasterio is not available.")
        with rasterio.open(self.filepath) as ds:
            data = ds.read()  # shape: (bands, height, width)
            self.num_bands = data.shape[0]
            if self.num_bands == 1:
                self.image_data = data[0]
            else:
                self.image_data = np.transpose(data, (1, 2, 0))
            self.metadata = ds.meta
            logging.info("Loaded image with rasterio.")
        return self.image_data

    def _load_with_glymur(self) -> np.ndarray:
        if glymur is None:
            raise ImportError("glymur is not available.")
        jp2 = glymur.Jp2k(self.filepath)
        data = jp2[:]  # read entire image
        if data.ndim == 2:
            self.image_data = data
            self.num_bands = 1
        else:
            self.image_data = data
            self.num_bands = data.shape[2] if data.ndim == 3 else data.shape[0]
        self.metadata = {'boxes': [str(box) for box in jp2.box]}
        logging.info("Loaded image with glymur.")
        return self.image_data

    def _load_with_gdal(self) -> np.ndarray:
        if gdal is None:
            raise ImportError("GDAL is not available.")
        ds = gdal.Open(self.filepath)
        if ds is None:
            raise IOError("GDAL could not open the file.")
        width = ds.RasterXSize
        height = ds.RasterYSize
        self.num_bands = ds.RasterCount
        bands = []
        for i in range(1, self.num_bands + 1):
            band = ds.GetRasterBand(i)
            bands.append(band.ReadAsArray())
        if self.num_bands > 1:
            self.image_data = np.stack(bands, axis=-1)
        else:
            self.image_data = bands[0]
        self.metadata = ds.GetMetadata()
        logging.info("Loaded image with GDAL.")
        return self.image_data

    def _load_with_spectral(self) -> np.ndarray:
        if spy is None:
            raise ImportError("Spectral Python (spy) is not available.")
        try:
            hdr_dict = spy.envi.read_envi_header(self.filepath)
        except Exception as e:
            logging.warning(f"Spectral header parsing failed, using custom parser: {e}")
            hdr_dict = _parse_envi_header_fixed(self.filepath)
        self.metadata = hdr_dict
        img = spy.envi.open(self.filepath)
        self.image_data = img.load()
        self.num_bands = 1 if self.image_data.ndim == 2 else self.image_data.shape[2]
        logging.info("Loaded image with Spectral Python.")
        return self.image_data

    def _load_raw_custom(self) -> np.ndarray:
        if exifread is None:
            raise ImportError("exifread is not available.")
        width, ok = QInputDialog.getInt(None, "Raw Image", "Enter image width:", 512, 1, 10000, 1)
        if not ok:
            raise Exception("Cancelled raw image loading.")
        height, ok = QInputDialog.getInt(None, "Raw Image", "Enter image height:", 512, 1, 10000, 1)
        if not ok:
            raise Exception("Cancelled raw image loading.")
        bands, ok = QInputDialog.getInt(None, "Raw Image", "Enter number of bands:", 1, 1, 10, 1)
        if not ok:
            raise Exception("Cancelled raw image loading.")
        bit_depth, ok = QInputDialog.getInt(None, "Raw Image", "Enter bit depth (8 or 16):", 8, 8, 16, 1)
        if not ok:
            raise Exception("Cancelled raw image loading.")

        dtype = np.uint8 if bit_depth == 8 else np.uint16
        expected_size = width * height * bands

        with open(self.filepath, 'rb') as f:
            data = f.read()
            f.seek(0)
            tags = exifread.process_file(f, details=False)

        self.metadata = {tag: str(value) for tag, value in tags.items()}
        arr = np.frombuffer(data, dtype=dtype)
        if arr.size != expected_size:
            raise IOError(f"Raw image size ({arr.size}) does not match expected ({expected_size}).")
        self.image_data = arr.reshape((height, width)) if bands == 1 else arr.reshape((height, width, bands))
        self.num_bands = bands
        logging.info(f"Loaded raw image: {width}x{height}, {bands} band(s), bit depth {bit_depth}.")
        return self.image_data

    def _load_with_h5py(self) -> np.ndarray:
        if h5py is None:
            raise ImportError("h5py is not available.")
        with h5py.File(self.filepath, 'r') as f:
            key = list(f.keys())[0]
            data = f[key][:]
            self.image_data = data
            if data.ndim == 2:
                self.num_bands = 1
            elif data.ndim == 3:
                self.num_bands = data.shape[2] if data.shape[-1] < data.shape[0] else data.shape[0]
            else:
                self.num_bands = 1
            self.metadata = dict(f.attrs)
            logging.info("Loaded image with h5py.")
        return self.image_data

    def _load_with_netcdf4(self) -> np.ndarray:
        if Dataset is None:
            raise ImportError("netCDF4 is not available.")
        ds = Dataset(self.filepath, 'r')
        for var_name in ds.variables:
            var = ds.variables[var_name]
            if var.ndim >= 2:
                data = var[:]
                self.image_data = data
                self.num_bands = 1 if data.ndim == 2 else (
                    data.shape[2] if data.shape[-1] < data.shape[0] else data.shape[0])
                self.metadata = dict(ds.__dict__)
                ds.close()
                logging.info("Loaded image with netCDF4.")
                return self.image_data
        ds.close()
        raise IOError("No suitable variable found in netCDF file.")

    def _load_with_h5py_l1r(self) -> np.ndarray:
        if h5py is None:
            raise ImportError("h5py is not available.")
        try:
            with h5py.File(self.filepath, 'r') as f:
                key = list(f.keys())[0]
                data = f[key][:]
                self.image_data = data
                if data.ndim == 2:
                    self.num_bands = 1
                elif data.ndim == 3:
                    self.num_bands = data.shape[2] if data.shape[-1] < data.shape[0] else data.shape[0]
                else:
                    self.num_bands = 1
                self.metadata = dict(f.attrs)
                logging.info("Loaded .l1r image with h5py.")
                return self.image_data
        except Exception as e:
            logging.error(f"h5py failed to open .l1r file: {e}")
            if gdal is not None:
                logging.info("Falling back to GDAL for .l1r file.")
                return self._load_with_gdal()
            else:
                raise IOError(f"Failed to load .l1r file with h5py and no GDAL fallback available: {e}")

    def _check_for_external_metadata(self) -> None:
        """
        Look for an external .hdr file associated with the image and merge its metadata.
        """
        base, _ = os.path.splitext(self.filepath)
        hdr_candidate = base + ".hdr"
        if os.path.exists(hdr_candidate):
            try:
                hdr_dict = spy.envi.read_envi_header(hdr_candidate) if spy else _parse_envi_header_fixed(hdr_candidate)
            except Exception as e:
                logging.warning(f"Failed to load header using spy.envi.read_envi_header: {e}. Using fixed parser.")
                hdr_dict = _parse_envi_header_fixed(hdr_candidate)
            self.metadata.update(hdr_dict)
            logging.info(f"Loaded external metadata from {hdr_candidate}")

    def get_band(self, band_index: int) -> np.ndarray:
        if self.image_data is None:
            raise ValueError("Image not loaded.")
        if self.num_bands == 1:
            return np.squeeze(self.image_data)
        if 0 <= band_index < self.num_bands:
            band = self.image_data[:, :, band_index]
            return np.ascontiguousarray(np.squeeze(band))
        else:
            raise IndexError("Band index out of range.")

    def get_pixel_spectrum(self, x: int, y: int) -> np.ndarray:
        """
        Return the spectrum (pixel values across all bands) at position (x, y).
        """
        if self.image_data is None:
            raise ValueError("Image not loaded.")
        if self.num_bands == 1:
            return np.array([self.image_data[y, x]])
        return self.image_data[y, x, :]


# -----------------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    # Update the path below to point to your file.
    # This could be a Landsat metadata file (.txt) or any supported image file.
    file_path = "path/to/your/file.txt"

    try:
        loader = ImageLoader(file_path)
        image = loader.load()
        logging.info(f"Loaded image with shape: {image.shape}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
