from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
import torch
from os import PathLike
import numpy as np
import torchvision.transforms as T

from ._auxiliaries import (
    add_coords_information,
    add_land_sea_mask,
    Coordinate,
    CoordinateBundle,
)

from ._chunkers import Chunker
import xarray
from enum import Enum

import glob
from pathlib import Path


# taken from our large 2008 dataset
T_MIN = 194.8391418457031
T_MAX = 327.346343994141
T_MEAN = 264.752433
T_STD = 18.535514


class DatasetDimension(Enum):
    """
    Enum for the dimension of the dataset.
    """

    DIM2 = 0
    DIM3 = 1


def _sample_values_to_3d_tensor(sample: xarray.DataArray) -> torch.Tensor:
    """Gets a sample from the dataset.

    Returns:
        A sample from the dataset at the given index. The sample is a 3D tensor.
        If a 2D sample is requested, a first dimension is added to the tensor as (1, x, y).
    """
    tensor = torch.Tensor(sample.values)
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)
    assert len(tensor.shape) == 3
    return tensor


def normalize(arr: torch.Tensor | np.ndarray):
    """Normalizes the array to be in the range [0, 1]. Uses
    the precomputed values from the year 2008 that might not be appropriate for other datasets.
    """
    return (arr - T_MIN) / (T_MAX - T_MIN)


def unnormalize(arr: torch.Tensor | np.ndarray):
    """Undoes the normalization of this package. Uses the precomputed values from the year 2008 that might not be appropriate for other datasets."""
    return arr * (T_MAX - T_MIN) + T_MIN


class Era5(Dataset):
    """Class that wraps the Era5 dataset to be used with PyTorch.
    Expect the dataset to be in NetCDF format, and contain latitude, longitude, level, and time dimensions.

    The samples can be accessed via an array-like access: dataset[0] will return the first sample.
    Increasing the index will first increase chunks (if applicable), then the level (if not fixed), then the time, unless a specific level has been fixed.

    The samples are returned as 2D or 3D tensors, depending on the chunker.
    If no chunker is set, we only return 2D tensors (as returning a full 3D sample is likely to be too large).

    Args:
        path: Path to the dataset.
        data_variable: Name of the variable in the dataset that contains the data, for example 't' for temperature.
        normalize: Whether to normalize the data to be in the range [0, 1]. For the temperature, uses precomputed values from the year 2008 that might not be appropriate for other datasets.
            The normalization also affects the passed coordinates. If a tuple is passed, uses the first number as min, the second as max.
            Defaults to False.
        normalize_coords: Whether to normalize the coordinates to be in the range [0, 1]. Defaults to True.
        level: If set, only samples from this level will be returned. If not set, all levels will be returned. This integer is the index of the level, not the actual value.
        chunker: If set, the dataset will be chunked into smaller samples. If not set, the dataset will be returned as is.
        transform: A function that generates a torch tensor, which can be consumed by a neural network, from an xarray sample. In the default transform,
            the sample is achieved by taking the values of the xarray sample and converting it to a 3D torch tensor. All of the sample transform should return
            a 3D Tensor and add auxiliary information (if applicable) along the first dimension.
        coords: The coordinates to add to the sample. Coordinates are added in the order that they are passed to the first dimension of the returned sample.
    """

    def __init__(
        self,
        dataset: str | PathLike | xarray.Dataset,
        data_variable: str,
        normalize: bool | tuple[float, float] = False,
        normalize_coords: bool = True,
        level: Optional[int] = None,
        chunker: Optional[Chunker] = None,
        coords: list[Coordinate] | CoordinateBundle = [],
        transform=None,
    ) -> None:
        self._era5: _Era5Base
        self._data_variable = data_variable
        self._normalize_coords = normalize_coords
        if chunker is None:
            self._era5 = _Era5Base(dataset, data_variable, level)
        else:
            self._era5 = _Era5Chunked(dataset, data_variable, chunker, level)
        self._normalize = normalize
        self._coords = coords.value if isinstance(coords, CoordinateBundle) else coords
        self.transform = transform

    def __str__(self) -> str:
        return str(self._era5)

    def __repr__(self) -> str:
        return self.__str__()

    def _normalize_sample(self, sample: xarray.DataArray) -> xarray.DataArray:
        """
        Normalizes the sample so that it falls between 0 and 1.
        """
        if isinstance(self._normalize, tuple):
            return (sample - self._normalize[0]) / (
                self._normalize[1] - self._normalize[0]
            )
        if self._normalize:
            return (sample - T_MIN) / (T_MAX - T_MIN)
        else:
            return sample

    def _get_normalized_data_array(self, index: int) -> xarray.DataArray:
        return self._normalize_sample(self._era5[index][self._data_variable])

    def __getitem__(self, index: int) -> torch.Tensor:
        """Gets a sample from the dataset.

        Returns:
            A sample from the dataset at the given index.
        """
        sample = add_coords_information(
            self._get_normalized_data_array(index),
            self._coords,
            normalize=self._normalize_coords,
        )
        return self.transform(sample) if self.transform is not None else sample

    def __len__(self) -> int:
        """
        Returns:
            Number of samples in this dataset.
        """
        return self._era5.__len__()

    @property
    def bytes(self) -> int:
        return self._era5.bytes

    @property
    def numel(self) -> int:
        return self._era5.numel


class Era5Lsm(Era5):
    """
    Era5 with a land-sea mask added in the first dimension.
    """

    def __init__(
        self,
        dataset: str | PathLike | xarray.Dataset,
        lsm_dataset: str | PathLike,
        data_variable: str,
        normalize: bool = True,
        level: Optional[int] = None,
        chunker: Optional[Chunker] = None,
    ) -> None:
        super().__init__(dataset, data_variable, normalize, level, chunker)
        if isinstance(lsm_dataset, xarray.Dataset):
            self._lsm_dataset = lsm_dataset
        else:
            self._lsm_dataset = xarray.open_dataset(lsm_dataset)  # type: ignore

    def __getitem__(self, index: int) -> torch.Tensor:
        return add_land_sea_mask(
            self._get_normalized_data_array(index), self._lsm_dataset
        )


class _Era5Base:
    """Class for a simple, unchunked access to Era5. See the Era5 class for more information."""

    def __init__(
        self,
        dataset: str | PathLike | xarray.Dataset,
        data_variable: str,
        level: Optional[int] = None,
    ) -> None:
        if isinstance(dataset, xarray.Dataset):
            self._dataset = dataset
        else:
            self._dataset = xarray.open_dataset(dataset)  # type: ignore
        if level is not None:
            # keeps dimension
            self._dataset = self._dataset.sel(level=[level], method="nearest")
        if data_variable not in self._dataset.data_vars:
            raise ValueError(f"'{data_variable}' not found in {self._dataset}.")
        self._no_level = "level" not in self._dataset.coords
        if self._no_level:
            self._num_levels = 1
        else:
            self._num_levels = self._dataset["level"].size
        self._num_timesteps = self._dataset["time"].size
        self._num_latitude = self._dataset["latitude"].size
        self._num_longitude = self._dataset["longitude"].size
        self._variable = data_variable
        self._data_variable = data_variable
        self._level = level
        self._dimension = DatasetDimension.DIM2

    def check_index(self, index):
        if index >= len(self):
            raise ValueError(f"Index {index} is larger than dataset ({len(self)})")

    def __str__(self) -> str:
        return f"Era5 Base: {self._dataset}"

    def __repr__(self) -> str:
        return self.__str__()

    def _isel_level_safe(self, level: int, **kwargs) -> xarray.Dataset:
        """
        Helper function to handle datasets more cleanly that don't have a level dimension.
        """
        if self._no_level:
            return self._dataset.isel(**kwargs)
        else:
            return self._dataset.isel(level=level, **kwargs)

    def __getitem__(self, index) -> xarray.Dataset:
        self.check_index(index)
        level_index = index % self._num_levels
        time_index = index // self._num_levels

        sample = self._isel_level_safe(time=time_index, level=level_index)
        return sample

    def __len__(self) -> int:
        return self._num_timesteps * self._num_levels

    @property
    def bytes(self) -> int:
        if self._dataset[self._data_variable].dtype == np.float32:
            el_size = 4
        elif self._dataset[self._data_variable].dtype == np.float64:
            el_size = 8
        else:
            raise ValueError(
                f"Unknown dtype {self._dataset[self._data_variable].dtype}"
            )
        return el_size * self.numel

    @property
    def numel(self) -> int:
        return int(self._dataset[self._data_variable].size)


class _Era5Chunked(_Era5Base):
    """Class for chunked access to Era5. See the Era5 class for more information.
    Chunking means that each sample is split into several chunks, that together makeup the full sample.

    The type of chunker determines the dimension of the samples.
    """

    def __init__(
        self,
        dataset: str | PathLike | xarray.Dataset,
        data_variable: str,
        chunker: Chunker,
        level: Optional[int] = None,
    ) -> None:
        super().__init__(dataset, data_variable, level)
        self._chunker = chunker
        if self._chunker.dimension == 2:
            self._sample_shape = {
                "latitude": self._num_latitude,
                "longitude": self._num_longitude,
            }
        elif self._chunker.dimension == 3:
            self._sample_shape = {
                "time": self._num_timesteps,
                "latitude": self._num_latitude,
                "longitude": self._num_longitude,
            }
        else:
            raise ValueError(f"Invalid dimension {self._chunker.dimension}")
        self._chunks_per_sample = self._chunker.get_num_chunks(self._sample_shape)

    def __str__(self) -> str:
        return f"Era5 Chunked ({self._chunker}): {self._dataset}"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, index) -> xarray.Dataset:
        self.check_index(index)
        # eventually the samples are too large to simply draw them, we have to work with the coordinates directly.
        sample_index = index // self._chunks_per_sample
        chunk_index = index % self._chunks_per_sample
        if self._chunker.dimension == 2:
            # pre-select time and level for the chunker, so it can only chunk over longitude and latitude
            level_index = sample_index % self._num_levels
            time_index = sample_index // self._num_levels
            sample = self._isel_level_safe(time=time_index, level=level_index)
        elif self._chunker.dimension == 3:
            # in this case, chunker has to deal with the time index
            level_index = sample_index
            sample = self._isel_level_safe(level=level_index)
        else:
            raise ValueError(f"Invalid dimension {self._chunker.dimension}")
        chunk = self._chunker.get_chunk_at(sample, chunk_index)
        return chunk

    def __len__(self) -> int:
        if self._chunker.dimension == 2:
            return self._num_timesteps * self._num_levels * self._chunks_per_sample
        elif self._chunker.dimension == 3:
            return self._num_levels * self._chunks_per_sample
        else:
            raise ValueError(f"Invalid dimension {self._chunker.dimension}")
