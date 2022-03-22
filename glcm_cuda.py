from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Tuple

import cupy as cp
import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from multikernel import glcm_module

MAX_VALUE_SUPPORTED = 256
NO_OF_VALUES_SUPPORTED = 256 ** 2
MAX_RADIUS_SUPPORTED = 127

MAX_THREADS = 512  # Lowest Maximum supported threads.

NO_OF_FEATURES = 8

PARTITION_SIZE = 10000


@dataclass
class GLCM:
    """
    
    Args:
        max_value: Maximum value of the image, default 256
        step_size: Step size of the window
        radius: Radius of the windows
        bins: Bin reduction. If None, then no reduction is done

    """

    step_size: int = 1
    radius: int = 2
    bins_from: int = 256
    bins: int = 256

    threads = MAX_VALUE_SUPPORTED + 1

    HOMOGENEITY = 0
    CONTRAST = 1
    ASM = 2
    MEAN_I = 3
    MEAN_J = 4
    VAR_I = 5
    VAR_J = 6
    CORRELATION = 7

    @property
    def diameter(self):
        return self.radius * 2 + 1

    @property
    def no_of_values(self):
        return self.diameter ** 2

    def __post_init__(self):
        self.i_flat = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)
        self.j_flat = cp.zeros((self.diameter ** 2,), dtype=cp.uint8)

        if not 0 <= self.radius < MAX_RADIUS_SUPPORTED:
            f"Radius {self.radius} should be in [0, {MAX_RADIUS_SUPPORTED})"
        if not (2 <= self.bins <= MAX_VALUE_SUPPORTED):
            raise ValueError(
                f"Bins {self.bins} should be in [2, {MAX_VALUE_SUPPORTED}]. "
            )
        if not 1 <= self.step_size:
            raise ValueError(
                f"Step Size {self.step_size} should be >= 1"
                f"If bins == 256, just use None."
            )

        self.glcm_0 = glcm_module.get_function('glcm_0')
        self.glcm_1 = glcm_module.get_function('glcm_1')
        self.glcm_2 = glcm_module.get_function('glcm_2')
        self.glcm_3 = glcm_module.get_function('glcm_3')

        os.environ['CUPY_EXPERIMENTAL_SLICE_COPY'] = '1'

    @staticmethod
    def binarize(im: np.ndarray, from_bins: int, to_bins: int) -> np.ndarray:
        """ Binarize an image from a certain bin to another

        Args:
            im: Image as np.ndarray
            from_bins: From the Bin of input image
            to_bins: To the Bin of output image

        Returns:
            Binarized Image

        """
        return (im.astype(np.float32) / from_bins * to_bins).astype(np.uint8)

    def from_3dimage(self,
                     im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a multi band image

        Args:
            im: A 3 dim image as an ndarray

        Returns:
            The GLCM Array 4dim with shape
                rows, cols, channel, feature
        """

        glcm_chs = []
        for ch in range(im.shape[-1]):
            glcm_chs.append(self.from_2dimage(im[..., ch]))

        return np.stack(glcm_chs, axis=2)

    def from_2dimage(self,
                     im: np.ndarray) -> np.ndarray:
        """ Generates the GLCM from a single band image

        Notes:
            This will actively partition the processing by blocks
            of PARTITION_SIZE.
            This allows for a reduction in GLCM creation.

        Args:
            im: Image in np.ndarray. Cannot be in cp.ndarray

        Returns:
            The GLCM Array 3dim with shape
                rows, cols, feature
        """

        windows_i, windows_j = \
            self.make_windows(im, self.diameter, self.step_size)

        glcm_features = cp.zeros(
            (windows_i.shape[0], NO_OF_FEATURES),
            dtype=cp.float32
        )

        windows_count = windows_i.shape[0]
        for partition in tqdm(
            range(math.ceil(windows_count / PARTITION_SIZE))):
            windows_part_i = windows_i[
                             (start := partition * PARTITION_SIZE):
                             (end := (partition + 1) * PARTITION_SIZE)
                             ]
            windows_part_j = windows_j[start:end]
            glcm_features[start:start + windows_part_i.shape[0]] = \
                self._from_windows(
                    windows_part_i,
                    windows_part_j
                )

        return glcm_features.reshape(
            im.shape[0] - self.radius * 2 - self.step_size,
            im.shape[1] - self.radius * 2 - self.step_size,
            NO_OF_FEATURES
        )

    def _from_windows(self,
                      i: np.ndarray,
                      j: np.ndarray):
        """ Generate the GLCM from the I J Window

        Examples:

            >>> ar_0 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> ar_1 = np.random.randint(0, 100, 10, dtype=np.uint8)
            >>> g = GLCM()._from_windows(ar_0[...,np.newaxis], ar_1[...,np.newaxis])

        Notes:
            i must be the same shape as j

        Args:
            i: I Window
            j: J Window

        Returns:
            The GLCM array, of size (8,)

        """
        assert i.ndim == 2 and j.ndim == 2, \
            "The input dimensions must be 2. " \
            "The 1st Dim is the partitioned windows flattened, " \
            "The 2nd is the window cells flattened"

        self.glcm = cp.zeros(
            (i.shape[0], self.bins, self.bins),
            dtype=cp.uint8
        )
        self.features = cp.zeros(
            (i.shape[0], 8),
            dtype=cp.float32
        )
        assert i.shape == j.shape, \
            f"Shape of i {i.shape} != j {j.shape}"

        i = self.binarize(i, self.bins_from, self.bins)
        j = self.binarize(j, self.bins_from, self.bins)

        no_of_windows = i.shape[0]

        if i.dtype != np.uint8 or j.dtype != np.uint8:
            raise ValueError(
                f"Image dtype must be np.uint8,"
                f" i: {i.dtype} j: {j.dtype}"
            )
        self.i_flat = cp.asarray(i)
        self.j_flat = cp.asarray(j)
        grid = self.calculate_grid(i.shape[0], self.bins)
        with cp.cuda.Device() as dev:
            dev.use()
            self.glcm_0(
                grid=grid,
                block=(MAX_THREADS,),
                args=(
                    self.i_flat,
                    self.j_flat,
                    self.bins,
                    self.no_of_values,
                    no_of_windows,
                    self.glcm,
                    self.features
                )
            )
            self.glcm_1(
                grid=grid,
                block=(MAX_THREADS,),
                args=(
                    self.glcm,
                    self.bins,
                    self.no_of_values,
                    no_of_windows,
                    self.features
                )
            )
            self.glcm_2(
                grid=grid,
                block=(MAX_THREADS,),
                args=(
                    self.glcm,
                    self.bins,
                    self.no_of_values,
                    no_of_windows,
                    self.features
                )
            )
            self.glcm_3(
                grid=grid,
                block=(MAX_THREADS,),
                args=(
                    self.glcm,
                    self.bins,
                    self.no_of_values,
                    no_of_windows,
                    self.features
                )
            )
            dev.synchronize()

        return self.features[:no_of_windows]

    @staticmethod
    def calculate_grid(
        window_count: int,
        glcm_size: int,
        thread_per_block: int = MAX_THREADS
    ) -> Tuple[int, int]:
        """ Calculates the required grid size

        Notes:
            There's 2 points where the number of threads

        Args:
            window_count:
            glcm_size:

        Returns:

        """
        blocks_req_glcm_calc = window_count * glcm_size * glcm_size / thread_per_block
        blocks_req_glcm_populate = glcm_size * window_count  # Do we need to div by thread_per_block?
        blocks_req = max(blocks_req_glcm_calc, blocks_req_glcm_populate)
        return (_ := int(blocks_req ** 0.5) + 1), _

    @staticmethod
    def make_windows(im: np.ndarray,
                     diameter: int,
                     step_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """ From a 2D image np.ndarray, convert it into GLCM IJ windows.

        Examples:

            Input 4 x 4 image. Radius = 1. Step Size = 1.

            +-+-+-+-+         +-----+
            |       |         | +---+-+
            |       |  ---->  | |   | |
            |       |         | |   | |
            |       |         +-+---+ |
            +-+-+-+-+           +-----+      flat
              4 x 4           1 x 1 x 3 x 3  ----> 1 x 9
                              +---+   +---+
                              flat    flat

            The output will be flattened on the x,y,

            Input 5 x 5 image. Radius = 1. Step Size = 1.

            1-2-+-+-+-+         1-----+    2-----+
            3 4       |         | +---+-+  | +---+-+  3-----+    4-----+
            |         |  ---->  | |   | |  | |   | |  | +---+-+  | +---+-+
            |         |         | |   | |  | |   | |  | |   | |  | |   | |
            |         |         +-+---+ |  +-+---+ |  | |   | |  | |   | |
            |         |           +-----+    +-----+  +-+---+ |  +-+---+ |
            +-+-+-+-+-+                                 +-----+    +-----+
              4 x 4                         2 x 2 x 3 x 3 ----> 4 x 9
                                            +---+   +---+ flat
                                            flat    flat
        Args:
            im: Input Image
            diameter: Diameter of Window
            step_size: Step Size between ij pairs

        Returns:
            The windows I, J suitable for GLCM.
            The first dimension: xy flat window indexes,
            the last dimension: xy flat indexes within each window.

        """
        # This will yield a shape (window_i, window_j, row, col)
        # E.g. 100x100 with 5x5 window -> 96, 96, 5, 5

        if im.shape[0] - step_size - diameter + 1 <= 0 or \
            im.shape[1] - step_size - diameter + 1 <= 0:
            raise ValueError(
                f"Step Size & Diameter exceeds size for windowing. "
                f"im.shape[0] {im.shape[0]} "
                f"- step_size {step_size} "
                f"- diameter{diameter} + 1 <= 0 or"
                f"im.shape[1] {im.shape[1]} "
                f"- step_size {step_size} "
                f"- diameter{diameter} + 1 <= 0 was not satisfied."
            )

        ij = view_as_windows(im, (diameter, diameter))
        i: np.ndarray = ij[:-step_size, :-step_size]
        j: np.ndarray = ij[step_size:, step_size:]

        i = i.reshape((-1, i.shape[-2], i.shape[-1])) \
            .reshape((i.shape[0] * i.shape[1], -1))
        j = j.reshape((-1, j.shape[-2], j.shape[-1])) \
            .reshape((j.shape[0] * j.shape[1], -1))

        return i, j
