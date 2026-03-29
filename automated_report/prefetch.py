"""
Simple Prefetch Module for Microscopy Image Pipeline.

This module provides a simple prefetching mechanism that can be easily
integrated into the existing report_generator.py with minimal changes.

Usage:
    from prefetch import ImagePrefetcher
    
    prefetcher = ImagePrefetcher(czi_files, cache_dir, prefetch_count=2)
    
    for czi_path in czi_files:
        # This returns immediately if prefetched, or loads synchronously if not
        max_proj, czi = prefetcher.get(czi_path)
        
        # GPU work happens here while next images load in background
        mask = segment_nuclei(max_proj[nuclear_channel], ...)
        
    prefetcher.shutdown()

Expected improvement: ~30-40% reduction in total time by hiding I/O latency.
"""

import os
import gc
import threading
import queue
import uuid
from pathlib import Path
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future
import time

import numpy as np
import tifffile


@dataclass
class PrefetchedImage:
    """Container for a prefetched image and its metadata."""
    max_proj: np.ndarray      # CYX array
    czi_obj: Any              # CziImage object
    load_time: float          # How long it took to load
    from_cache: bool          # Whether it came from cache


class ImagePrefetcher:
    """
    Asynchronous image prefetcher that loads images ahead of processing.
    
    This class loads and preprocesses images in background threads while
    the main thread runs GPU segmentation, hiding I/O latency.
    
    Thread-safe and handles HDD RAID arrays well with multiple I/O threads.
    """
    
    def __init__(
        self,
        czi_files: list[str],
        cache_dir: str,
        prefetch_count: int = 2,
        io_threads: int = 2,
        use_cache: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize the prefetcher.
        
        Args:
            czi_files: Ordered list of CZI file paths to process
            cache_dir: Directory for cached max projections
            prefetch_count: Number of images to load ahead
            io_threads: Number of I/O threads (2 is good for HDDs)
            use_cache: Whether to use/create cached max projections
            verbose: Print status messages
        """
        self.czi_files = czi_files
        self.cache_dir = cache_dir
        self.prefetch_count = prefetch_count
        self.use_cache = use_cache
        self.verbose = verbose
        
        # Create cache directory
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
        
        # Thread pool for I/O operations
        self.executor = ThreadPoolExecutor(max_workers=io_threads)
        
        # Cache of prefetched images: path -> Future[PrefetchedImage]
        self.prefetch_cache: dict[str, Future] = {}
        self.cache_lock = threading.Lock()
        
        # Track current position in file list
        self.current_index = 0
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'prefetch_hits': 0,
            'prefetch_misses': 0,
            'total_io_time': 0.0,
            'total_wait_time': 0.0,
        }
        
        # Start prefetching first few images
        self._schedule_prefetch()
    
    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Prefetch] {msg}")

    @staticmethod
    def _is_valid_max_projection(arr: np.ndarray) -> bool:
        """Validate cached max projection shape before using it."""
        return arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0 and arr.shape[2] > 0

    @staticmethod
    def _write_tiff_atomic(path: str, data: np.ndarray):
        """Write TIFF atomically to avoid partial cache files."""
        tmp_path = f"{path}.tmp.{uuid.uuid4().hex}"
        try:
            tifffile.imwrite(
                tmp_path,
                data,
                compression='zstd',
                compressionargs={'level': 3},
            )
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
    
    def _load_image(self, czi_path: str) -> PrefetchedImage:
        """
        Load a single image (runs in thread pool).
        
        Returns cached max projection if available, otherwise loads CZI
        and computes max projection.
        """
        # Import here to avoid issues with thread-local state
        from .parse_czi import CziImage, microscopy_image_stem
        
        start_time = time.time()
        filename = microscopy_image_stem(czi_path)
        cache_file = os.path.join(self.cache_dir, f"{filename}_max.tif")
        
        # Check cache first
        if self.use_cache and os.path.exists(cache_file):
            try:
                max_proj = tifffile.imread(cache_file)
                if not self._is_valid_max_projection(max_proj):
                    raise ValueError(f"invalid cached projection shape: {max_proj.shape}")

                czi = CziImage(czi_path)  # Still need metadata

                load_time = time.time() - start_time
                self._log(f"Loaded cached: {filename} ({load_time:.1f}s)")

                return PrefetchedImage(
                    max_proj=max_proj,
                    czi_obj=czi,
                    load_time=load_time,
                    from_cache=True,
                )
            except Exception as e:
                self._log(f"Corrupt cache for {filename}, rebuilding ({e})")
                try:
                    os.remove(cache_file)
                except OSError:
                    pass
        
        # Load full CZI and compute max projection
        czi = CziImage(czi_path)
        max_proj = czi.zmax()
        if not self._is_valid_max_projection(max_proj):
            raise ValueError(f"computed invalid max projection shape: {max_proj.shape}")
        
        # Cache the result
        if self.use_cache:
            self._write_tiff_atomic(cache_file, max_proj)
        
        # Clear raw data to save memory
        if hasattr(czi, '_image_data'):
            czi._image_data = None
        gc.collect()
        
        load_time = time.time() - start_time
        self._log(f"Loaded & projected: {filename} ({load_time:.1f}s)")
        
        return PrefetchedImage(
            max_proj=max_proj,
            czi_obj=czi,
            load_time=load_time,
            from_cache=False,
        )
    
    def _schedule_prefetch(self):
        """Schedule loading of upcoming images."""
        with self.cache_lock:
            # Determine which files to prefetch
            start_idx = self.current_index
            end_idx = min(start_idx + self.prefetch_count + 1, len(self.czi_files))
            
            for idx in range(start_idx, end_idx):
                path = self.czi_files[idx]
                
                # Skip if already prefetching
                if path in self.prefetch_cache:
                    continue
                
                # Submit load task
                future = self.executor.submit(self._load_image, path)
                self.prefetch_cache[path] = future
                self._log(f"Scheduled prefetch: {Path(path).name}")
    
    def get(self, czi_path: str) -> Tuple[np.ndarray, Any]:
        """
        Get the max projection and CZI object for a file.
        
        If the image was prefetched, returns immediately.
        Otherwise, loads synchronously and schedules more prefetches.
        
        Args:
            czi_path: Path to the CZI file
            
        Returns:
            (max_proj, czi_object) tuple
        """
        wait_start = time.time()
        
        with self.cache_lock:
            future = self.prefetch_cache.get(czi_path)
        
        if future is not None:
            # Wait for prefetch to complete
            result = future.result()
            wait_time = time.time() - wait_start
            
            self.stats['prefetch_hits'] += 1
            self.stats['total_io_time'] += result.load_time
            self.stats['total_wait_time'] += wait_time
            
            if result.from_cache:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1
            
            # Clean up and schedule more
            with self.cache_lock:
                if czi_path in self.prefetch_cache:
                    del self.prefetch_cache[czi_path]
                self.current_index = self.czi_files.index(czi_path) + 1
            
            self._schedule_prefetch()
            
            return result.max_proj, result.czi_obj
        
        # Not prefetched - load synchronously (shouldn't happen often)
        self._log(f"WARN: Synchronous load (not prefetched): {Path(czi_path).name}")
        self.stats['prefetch_misses'] += 1
        
        result = self._load_image(czi_path)
        self.stats['total_io_time'] += result.load_time
        
        if result.from_cache:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
        
        with self.cache_lock:
            self.current_index = self.czi_files.index(czi_path) + 1
        
        self._schedule_prefetch()
        
        return result.max_proj, result.czi_obj
    
    def shutdown(self):
        """Shutdown the prefetcher and clean up resources."""
        # Cancel pending prefetches
        with self.cache_lock:
            for future in self.prefetch_cache.values():
                future.cancel()
            self.prefetch_cache.clear()
        
        self.executor.shutdown(wait=False)
    
    def print_stats(self):
        """Print prefetching statistics."""
        print("\n" + "=" * 50)
        print("PREFETCH STATISTICS")
        print("=" * 50)
        print(f"  Prefetch hits:  {self.stats['prefetch_hits']}")
        print(f"  Prefetch misses: {self.stats['prefetch_misses']}")
        print(f"  Cache hits:     {self.stats['cache_hits']}")
        print(f"  Cache misses:   {self.stats['cache_misses']}")
        print(f"  Total I/O time: {self.stats['total_io_time']:.1f}s")
        print(f"  Total wait time: {self.stats['total_wait_time']:.1f}s")
        
        if self.stats['prefetch_hits'] + self.stats['prefetch_misses'] > 0:
            effective_rate = self.stats['prefetch_hits'] / (
                self.stats['prefetch_hits'] + self.stats['prefetch_misses']
            )
            print(f"  Prefetch effectiveness: {effective_rate:.1%}")
        print("=" * 50)


class GPUBatchAccumulator:
    """
    Accumulate similarly-sized images for batch GPU processing.
    
    Cellpose can process multiple images in a batch if they have similar
    dimensions, which improves GPU utilization.
    """
    
    def __init__(
        self,
        max_batch_size: int = 4,
        size_tolerance: float = 0.1,  # 10% size difference allowed
    ):
        self.max_batch_size = max_batch_size
        self.size_tolerance = size_tolerance
        self.pending: list[tuple[np.ndarray, str, Any]] = []  # (image, name, metadata)
    
    def add(self, image: np.ndarray, name: str, metadata: Any = None) -> Optional[list]:
        """
        Add an image to the batch.
        
        Returns a batch if ready for processing, None otherwise.
        """
        if not self.pending:
            self.pending.append((image, name, metadata))
            return None
        
        # Check if this image is compatible with pending batch
        ref_shape = self.pending[0][0].shape
        shape_compatible = (
            abs(image.shape[0] - ref_shape[0]) / ref_shape[0] <= self.size_tolerance and
            abs(image.shape[1] - ref_shape[1]) / ref_shape[1] <= self.size_tolerance
        )
        
        if shape_compatible and len(self.pending) < self.max_batch_size:
            self.pending.append((image, name, metadata))
            return None
        
        # Return current batch and start new one
        batch = self.pending
        self.pending = [(image, name, metadata)]
        return batch
    
    def flush(self) -> Optional[list]:
        """Return any remaining images as a final batch."""
        if self.pending:
            batch = self.pending
            self.pending = []
            return batch
        return None
