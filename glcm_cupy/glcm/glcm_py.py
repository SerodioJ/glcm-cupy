from dataclasses import dataclass

import numpy as np
from skimage.util import view_as_windows
from tqdm import tqdm

from glcm_cupy.conf import NO_OF_FEATURES
from glcm_cupy.glcm_py_base import GLCMPyBase


def glcm_py_3d(ar: np.ndarray, bin_from: int, bin_to: int,
               radius: int = 2,
               step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_3d(ar)


def glcm_py_2d(ar: np.ndarray,
               bin_from: int,
               bin_to: int,
               radius: int = 2,
               step: int = 1):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to,
                  radius=radius,
                  step=step).glcm_2d(ar)


def glcm_py_ij(i: np.ndarray,
               j: np.ndarray,
               bin_from: int, bin_to: int):
    return GLCMPy(bin_from=bin_from,
                  bin_to=bin_to).glcm_ij(i, j)


@dataclass
class GLCMPy(GLCMPyBase):
    step: int = 1

    def glcm_2d(self, ar: np.ndarray):
        ar = (ar / self.bin_from * self.bin_to).astype(np.uint8)
        ar_w = view_as_windows(ar, (self.diameter, self.diameter))

        def flat(ar: np.ndarray):
            ar = ar.reshape((-1, self.diameter, self.diameter))
            return ar.reshape((ar.shape[0], -1))

        ar_w_i = flat(ar_w[self.step:-self.step, self.step:-self.step])
        ar_w_j_sw = flat(ar_w[self.step * 2:, :-self.step * 2])
        ar_w_j_s = flat(ar_w[self.step * 2:, self.step:-self.step])
        ar_w_j_se = flat(ar_w[self.step * 2:, self.step * 2:])
        ar_w_j_e = flat(ar_w[self.step:-self.step, self.step * 2:])

        feature_ar = np.zeros((ar_w_i.shape[0], 4, NO_OF_FEATURES))

        for j_e, ar_w_j in enumerate(
            (ar_w_j_sw, ar_w_j_s, ar_w_j_se, ar_w_j_e)):
            for e, (i, j) in tqdm(enumerate(zip(ar_w_i, ar_w_j)),
                                  total=ar_w_i.shape[0]):
                feature_ar[e, j_e] = self.glcm_ij(i, j)

        h, w = ar_w.shape[:2]
        feature_ar = feature_ar.mean(axis=1)

        feature_ar = feature_ar.reshape(
            (h - self.step * 2, w - self.step * 2, NO_OF_FEATURES))

        return self.normalize_features(feature_ar)

    def glcm_3d(self, ar: np.ndarray):

        return np.stack([self.glcm_2d(ar[..., ch])
                         for ch
                         in range(ar.shape[-1])], axis=2)
