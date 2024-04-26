from __future__ import annotations
from typing import Optional, Hashable, Mapping
from enum import Enum
import xarray
from abc import ABC, abstractmethod
import numpy as np


class Offset(Enum):
    """Enum for the offset of the chunks."""

    NONE = 0
    RANDOM = 1


class Chunker(ABC):
    """Class that takes in image data and returns a chunked version of it.

    All of the chunks together make up one full image.

    Args:
        chunk_shape: The shape of the chunks. Might be 2D or 3D. Alternatively, a dictionary may be passed,
        which contains the keys latitude, longitude and optionally time, and as values the chunk shapes of the corresponding
        dimension
        padding: The padding type to use. Currently only NONE is implemented.
    """

    def __init__(self, chunk_shape: tuple | dict) -> None:
        self._chunk_shape: dict

        if isinstance(chunk_shape, tuple):
            if len(chunk_shape) == 2:
                self._chunk_shape = {
                    "latitude": chunk_shape[0],
                    "longitude": chunk_shape[1],
                }
            elif len(chunk_shape) == 3:
                self._chunk_shape = {
                    "time": chunk_shape[0],
                    "latitude": chunk_shape[1],
                    "longitude": chunk_shape[2],
                }
            else:
                raise ValueError(
                    f"Wrong length passed for chunk shape (must be 2 or 3): {chunk_shape}"
                )
        elif isinstance(chunk_shape, dict):
            self._chunk_shape = chunk_shape
        else:
            raise ValueError(f"Wrong type passed for chunk shape: {type(chunk_shape)}")

    def __str__(self) -> str:
        return f"Chunker {self._chunk_shape}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def dimension(self) -> int:
        """
        Returns:
            The dimension of the chunks, i.e. 2 or 3.
        """
        return len(self._chunk_shape.values())

    def _check_sample_validity(self, sample: xarray.Dataset, idx: int):
        if self.dimension == len(sample.coords.keys()):
            raise ValueError(
                f"Sample must be {self.dimension}D, but is {len(sample.coords.keys())}D"
            )
        if idx >= self.get_num_chunks(sample.shape):
            raise ValueError(
                f"Index {idx} out of bounds, max index is {self.get_num_chunks(sample.shape) - 1}"
            )

    @abstractmethod
    def get_chunk_at(self, sample: xarray.Dataset, idx: int) -> xarray.Dataset:
        """Returns the chunk at the given index for the sample.

        Args:
            sample: The sample to get the chunk from.
            idx: The index of the chunk to get.
        """
        raise NotImplementedError()

    @abstractmethod
    # convoluted type hinting for sample shape is required, since xarray.Dataset.coords.dims is a Mapping[Hashable, int], which
    # is invariant in its first type
    def get_num_chunks(
        self, sample_shape: dict[str, int] | xarray.Dataset | Mapping[Hashable, int]
    ) -> int:
        """Returns the number of possible chunks in a sample of the given shape.

        Args:
            sample_shape: The shape of the sample.
        """
        raise NotImplementedError()


class ContinuousChunker(Chunker):
    """Chunker that chunks the image into a continuous grid of chunks. The chunks don't overlap,
    and they make up the full image together.
    """

    def __init__(
        self,
        chunk_shape: tuple | dict,
        offset: Offset = Offset.NONE,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(chunk_shape)
        self._offset = None
        self._offset_mode = offset
        self._seed = seed

    def _determine_offset(self, sample: xarray.Dataset):
        self._offset = {}
        if self._offset_mode == Offset.RANDOM:
            self.rng = np.random.default_rng(self._seed)
            # we had to defer the off set generation to here, because we don't know the sample shape yet
            for coord_name, coord_size in sample.coords.dims.items():
                self._offset[coord_name] = np.random.randint(coord_size)
        elif self._offset_mode == Offset.NONE:
            for coord_name, coord_size in sample.coords.dims.items():
                self._offset[coord_name] = 0
        else:
            raise ValueError(f"Unknown offset mode {self._offset_mode}")

    def get_chunk_at(self, sample: xarray.Dataset, idx: int) -> xarray.Dataset:
        # this has to be done in a lazy fashion, because we don't know the sample shape yet
        if self._offset is None:
            self._determine_offset(sample)
        assert self._offset is not None

        for coord_name, coord_value in self._chunk_shape.items():
            coordinate = sample[coord_name]
            dim_size = coordinate.size
            offset = self._offset[coord_name]

            num_entries = dim_size // coord_value
            if num_entries == 0:  # empty coordinate
                continue
            idx, current = divmod(idx, num_entries)

            start = (current * coord_value + offset) % dim_size
            end = ((current + 1) * coord_value + offset) % dim_size

            start_val = coordinate[start]
            end_val = coordinate[end]

            if end == 0:  # else the wrap around is incorrect
                end = dim_size

            bigger_equal_start = coordinate >= start_val
            smaller_than_end = coordinate < end_val

            if start < end:
                sample = sample.isel({coord_name: slice(start, end)})
            else:  # wrap-around case
                sample = sample.isel(
                    {coord_name: bigger_equal_start | smaller_than_end}
                )
        return sample

    def get_num_chunks(
        self, sample_shape: dict[str, int] | xarray.Dataset | Mapping[Hashable, int]
    ) -> int:
        if isinstance(sample_shape, xarray.Dataset):
            sample_shape = sample_shape.coords.dims
        num_chunks = 1
        for coord_name, coord_size in sample_shape.items():
            if coord_name not in self._chunk_shape:
                continue
            assert isinstance(
                coord_name, str
            )  # it should normally be a str anyway, unless you do weird things
            num_chunks *= coord_size // self._chunk_shape[coord_name]
        return num_chunks


class SingleRandomWrapChunker(ContinuousChunker):
    """Chunker that returns a random chunker per image. We assume that the data lives on a spherical
    surface and that its coordinates thus wrap around (e.g. coords[0] is adjacent to coords[-1]).
    The returned chunk may wrap around coordinate borders.
    """

    def __init__(self, chunk_shape: tuple, seed: Optional[int] = None) -> None:
        super().__init__(chunk_shape, offset=Offset.RANDOM)

    def get_chunk_at(self, sample: xarray.Dataset, idx: int) -> xarray.Dataset:
        assert idx == 0
        return super().get_chunk_at(sample, idx)

    def get_num_chunks(self, _) -> int:
        return 1


class SingleChunker(Chunker):
    """Simple Chunker that returns only a single chunk per image. See the base class for more information.

    Args:
        offset: The offset of the chunk. If None, the chunk will start at 0 in all dimensions. If specified,
        the offset must have the same length as the chunk shape.
    """

    def __init__(
        self,
        chunk_shape: tuple[int, ...],
        offset: Optional[list[int] | dict] = None,
    ) -> None:
        super().__init__(chunk_shape)
        if offset is None:
            self._offset = {k: 0 for k, _ in self._chunk_shape.items()}
        elif isinstance(offset, list):
            if len(offset) == 2:
                self._offset = {"latitude": offset[0], "longitude": offset[1]}
            elif len(offset) == 3:
                self._offset = {
                    "time": offset[0],
                    "latitude": offset[1],
                    "longitude": offset[2],
                }
        else:
            self._offset = offset

    def get_chunk_at(self, sample: xarray.Dataset, idx: int) -> xarray.Dataset:
        if idx != 0:
            raise ValueError(f"Index {idx} out of bounds, max index is 0")
        for coord_name, coord_value in self._chunk_shape.items():
            coord_offset = self._offset[coord_name]
            sample = sample.isel(
                {coord_name: slice(coord_offset, coord_value + coord_offset)}
            )
        return sample

    def get_num_chunks(self, _) -> int:
        return 1
