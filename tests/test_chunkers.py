from era5_torch import (
    ContinuousChunker,
    SingleChunker,
    SingleRandomWrapChunker,
    Offset,
)
import pytest
import numpy as np
import xarray as xr


# fixtures
@pytest.fixture(scope="session")
def arr():
    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                ]
            ]
        ]
    )
    ds = xr.Dataset(
        data_vars={
            "t": (
                ["level", "time", "latitude", "longitude"],
                data,
            )
        },
        coords={
            "level": [1000],
            "time": [0],
            "latitude": np.arange(4),
            "longitude": np.arange(6),
        },
    )
    return ds


@pytest.fixture(scope="session")
def arr3d():
    data = np.array(
        [
            [
                [
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                ],
                [
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36],
                    [37, 38, 39, 40, 41, 42],
                    [43, 44, 45, 46, 47, 48],
                ],
                [
                    [49, 50, 51, 52, 53, 54],
                    [55, 56, 57, 58, 59, 60],
                    [61, 62, 63, 64, 65, 66],
                    [67, 68, 69, 70, 71, 72],
                ],
                [
                    [73, 74, 75, 76, 77, 78],
                    [79, 80, 81, 82, 83, 84],
                    [85, 86, 87, 88, 89, 90],
                    [91, 92, 93, 94, 95, 96],
                ],
            ]
        ]
    )
    # make an xarray dataset with the data and dimensions level time latitude longitude
    ds = xr.Dataset(
        data_vars={"t": (["level", "time", "latitude", "longitude"], data)},
        coords={
            "level": [1000],
            "time": np.arange(4),
            "latitude": np.arange(4),
            "longitude": np.arange(6),
        },
    )
    return ds


def test_chunks_2d(arr):
    chunk_shape = (2, 3)
    chunker = ContinuousChunker(chunk_shape)
    assert np.allclose(
        chunker.get_chunk_at(arr, 0)["t"].values, np.array([[1, 2, 3], [7, 8, 9]])
    )
    assert np.allclose(
        chunker.get_chunk_at(arr, 1)["t"].values,
        np.array([[13, 14, 15], [19, 20, 21]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr, 2)["t"].values,
        np.array([[4, 5, 6], [10, 11, 12]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr, 3)["t"].values,
        np.array([[16, 17, 18], [22, 23, 24]]),
    )
    assert chunker.get_num_chunks(arr) == 4


def test_chunks_3d(arr3d):
    chunk_shape = (2, 2, 3)
    chunker = ContinuousChunker(chunk_shape)
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 0)["t"].values,
        np.array([[[1, 2, 3], [7, 8, 9]], [[25, 26, 27], [31, 32, 33]]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 1)["t"].values,
        np.array([[[49, 50, 51], [55, 56, 57]], [[73, 74, 75], [79, 80, 81]]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 2)["t"].values,
        np.array([[[13, 14, 15], [19, 20, 21]], [[37, 38, 39], [43, 44, 45]]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 3)["t"].values,
        np.array([[[61, 62, 63], [67, 68, 69]], [[85, 86, 87], [91, 92, 93]]]),
    )
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 4)["t"].values,
        np.array([[[4, 5, 6], [10, 11, 12]], [[28, 29, 30], [34, 35, 36]]]),
    )
    assert chunker.get_num_chunks(arr3d) == 8


def test_single_chunker_2d(arr):
    chunk_shape = (2, 3)
    chunker = SingleChunker(chunk_shape)
    assert np.allclose(
        chunker.get_chunk_at(arr, 0)["t"].values, np.array([[1, 2, 3], [7, 8, 9]])
    )
    assert chunker.get_num_chunks(arr) == 1


def test_single_chunker_3d(arr3d):
    chunk_shape = (2, 2, 3)
    chunker = SingleChunker(chunk_shape)
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 0)["t"].values,
        np.array([[[1, 2, 3], [7, 8, 9]], [[25, 26, 27], [31, 32, 33]]]),
    )
    assert chunker.get_num_chunks(arr3d)


def test_single_chunker_throws(arr):
    chunker_shape = (2, 2, 3)
    chunker = SingleChunker(chunker_shape)
    with pytest.raises(ValueError):
        chunker.get_chunk_at(arr, 1)

    chunker_shape = (2, 2)
    chunker = SingleChunker(chunker_shape)
    with pytest.raises(ValueError):
        chunker.get_chunk_at(arr, 1)


def test_single_chunker_offset_2d(arr):
    chunk_shape = (2, 3)
    chunker = SingleChunker(chunk_shape, offset=[1, 1])
    assert np.allclose(
        chunker.get_chunk_at(arr, 0)["t"].values, np.array([[8, 9, 10], [14, 15, 16]])
    )
    assert chunker.get_num_chunks(arr) == 1


def test_single_chunker_offset_3d(arr3d):
    chunk_shape = (2, 2, 3)
    chunker = SingleChunker(chunk_shape, offset=[1, 1, 1])
    assert np.allclose(
        chunker.get_chunk_at(arr3d, 0)["t"].values,
        np.array([[[32, 33, 34], [38, 39, 40]], [[56, 57, 58], [62, 63, 64]]]),
    )
    assert chunker.get_num_chunks(arr3d) == 1


def test_random_chunker_2d(arr):
    chunk_shape = (2, 3)
    chunker = SingleRandomWrapChunker(chunk_shape, seed=0)
    assert chunker.get_num_chunks(arr) == 1
    arr_raw = arr["t"].values.squeeze()
    for _ in range(20):
        sample = chunker.get_chunk_at(arr, 0)["t"].values.squeeze()
        assert sample.shape == chunk_shape
        assert np.isin(sample, arr_raw).all()
        assert np.isin(sample, arr_raw).sum() == 6


def test_random_chunker_3d(arr3d):
    chunk_shape = (2, 2, 3)
    chunker = SingleRandomWrapChunker(chunk_shape, seed=0)
    arr_raw = arr3d["t"].values.squeeze()
    assert chunker.get_num_chunks(arr3d) == 1
    for _ in range(20):
        sample = chunker.get_chunk_at(arr3d, 0)["t"].values.squeeze()
        assert sample.shape == chunk_shape
        assert np.isin(sample, arr_raw).all()
        assert np.isin(sample, arr_raw).sum() == 12


def test_random_offset_continuous_chunker_2d(arr):
    chunk_shape = (2, 3)
    chunker = ContinuousChunker(chunk_shape, offset=Offset.RANDOM, seed=0)
    arr_raw = arr["t"].values.squeeze()
    assert chunker.get_num_chunks(arr) == 4
    mask = np.zeros(arr_raw.shape, dtype=bool)

    for i in range(4):
        sample = chunker.get_chunk_at(arr, i)["t"].values.squeeze()
        assert sample.shape == chunk_shape
        assert np.isin(sample, arr_raw).all()
        assert np.isin(sample, arr_raw).sum() == 6
        mask = np.logical_or(mask, np.isin(arr_raw, sample))

    assert mask.all()


def test_random_offset_continuous_chunker_3d(arr3d):
    chunk_shape = (2, 2, 3)
    chunker = ContinuousChunker(chunk_shape, offset=Offset.RANDOM, seed=0)
    arr_raw = arr3d["t"].values.squeeze()
    assert chunker.get_num_chunks(arr3d) == 8
    mask = np.zeros(arr_raw.shape, dtype=bool)

    for i in range(8):
        sample = chunker.get_chunk_at(arr3d, i)["t"].values.squeeze()
        assert sample.shape == chunk_shape
        assert np.isin(sample, arr_raw).all()
        assert np.isin(sample, arr_raw).sum() == 12
        mask = np.logical_or(mask, np.isin(arr_raw, sample))

    assert mask.all()
