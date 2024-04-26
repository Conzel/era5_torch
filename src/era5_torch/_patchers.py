from abc import ABC, abstractmethod
import torch
from enum import Enum


class PatchDimension(Enum):
    """
    Enum for specifying the dimension of a patch.
    """

    DIM2 = 1
    DIM3 = 2


class Patcher(ABC):
    """
    Class that extracts patches from an image.
    """

    def __init__(self, patch_shape: tuple) -> None:
        if len(patch_shape) == 2:
            self._patch_dimension = PatchDimension.DIM2
        elif len(patch_shape) == 3:
            self._patch_dimension = PatchDimension.DIM3
        else:
            raise ValueError("Patch must be 2D or 3D")
        self._patch_shape = patch_shape

    def _validate_sample_shape(self, sample: torch.Tensor) -> None:
        if len(sample.shape) != len(self._patch_shape):
            raise ValueError(f"Sample must be {len(self._patch_shape)}D")
        for i in range(len(sample.shape)):
            if sample.shape[i] < self._patch_shape[i]:
                raise ValueError(
                    f"Sample must be at least {self._patch_shape[i]} in dimension {i}"
                )

    @abstractmethod
    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BeginningPatcher(Patcher):
    """
    Returns a patch from the start of the array.
    """

    def __init__(self, patch_shape: tuple) -> None:
        super().__init__(patch_shape)

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        super()._validate_sample_shape(sample)
        if self._patch_dimension == PatchDimension.DIM2:
            patched_sample = sample[0 : self._patch_shape[0], 0 : self._patch_shape[1]]
        elif self._patch_dimension == PatchDimension.DIM3:
            patched_sample = sample[
                0 : self._patch_shape[0],
                0 : self._patch_shape[1],
                0 : self._patch_shape[2],
            ]
        else:
            raise ValueError("Patch must be 2D or 3D")
        assert patched_sample.shape == self._patch_shape
        return patched_sample
