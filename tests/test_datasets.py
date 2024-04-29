from era5_torch import (
    Era5,
    ContinuousChunker,
    Coordinate,
    CoordinateBundle,
    normalize,
    unnormalize,
)
import numpy as np
import torch
import xarray as xr
import pytest

temperature_path = "tests/era5_temperature_small.nc"
lsm_path = "tests/era5_lsm_small.nc"


def test_normalize_unnormalize_cancels():
    for i in range(100):
        xs = torch.rand(2, 3) * 100
        xs_prime = normalize(unnormalize(xs))
        xs_prime_reverse = unnormalize(normalize(xs))
        assert np.allclose(xs, xs_prime, atol=1e-4)
        assert np.allclose(xs_prime_reverse, xs_prime, atol=1e-4)
        assert np.allclose(xs_prime_reverse, xs, atol=1e-4)


def test_era5_retrieval_correct_values():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(temperature_path, "t", normalize=False)

    assert era5[0].squeeze().shape == (ds.latitude.size, ds.longitude.size)
    assert torch.allclose(
        era5[0].squeeze(), torch.Tensor(ds.t.isel(time=0, level=0).values)
    )
    assert torch.allclose(
        era5[1].squeeze(), torch.Tensor(ds.t.isel(time=0, level=1).values)
    )
    assert torch.allclose(
        era5[3].squeeze(), torch.Tensor(ds.t.isel(time=1, level=0).values)
    )


def test_era5_retrieval_correct_values_fixed_level():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(temperature_path, "t", normalize=False, level=1000)

    assert era5[0].squeeze().shape == (ds.latitude.size, ds.longitude.size)
    assert torch.allclose(
        era5[0].squeeze(), torch.Tensor(ds.t.isel(time=0, level=0).values)
    )
    assert torch.allclose(
        era5[1].squeeze(), torch.Tensor(ds.t.isel(time=1, level=0).values)
    )
    assert torch.allclose(
        era5[2].squeeze(), torch.Tensor(ds.t.isel(time=2, level=0).values)
    )


def test_era5_chunked_retrieval_correct_values_2d():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(
        temperature_path, "t", normalize=False, chunker=ContinuousChunker((7, 7))
    )

    assert torch.allclose(
        era5[0],
        torch.Tensor(
            ds.t.isel(
                time=0, level=0, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[1],
        torch.Tensor(
            ds.t.isel(
                time=0, level=0, latitude=slice(7, 14), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[3],
        torch.Tensor(
            ds.t.isel(
                time=0, level=0, latitude=slice(0, 7), longitude=slice(7, 14)
            ).values
        ),
    )
    assert torch.allclose(
        era5[9],
        torch.Tensor(
            ds.t.isel(
                time=0, level=1, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )


def test_era5_chunked_retrieval_correct_values_fixed_level():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(
        temperature_path,
        "t",
        normalize=False,
        chunker=ContinuousChunker((7, 7)),
        level=1,
    )

    assert torch.allclose(
        era5[0],
        torch.Tensor(
            ds.t.isel(
                time=0, level=2, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[1],
        torch.Tensor(
            ds.t.isel(
                time=0, level=2, latitude=slice(7, 14), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[3],
        torch.Tensor(
            ds.t.isel(
                time=0, level=2, latitude=slice(0, 7), longitude=slice(7, 14)
            ).values
        ),
    )
    assert torch.allclose(
        era5[9],
        torch.Tensor(
            ds.t.isel(
                time=1, level=2, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )


def test_era5_chunked_retrieval_correct_values_3d():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(
        temperature_path, "t", normalize=False, chunker=ContinuousChunker((2, 7, 7))
    )

    assert torch.allclose(
        era5[0],
        torch.Tensor(
            ds.t.isel(
                time=slice(0, 2), level=0, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[1],
        torch.Tensor(
            ds.t.isel(
                time=slice(2, 4), level=0, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[2],
        torch.Tensor(
            ds.t.isel(
                time=slice(4, 6), level=0, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[4],
        torch.Tensor(
            ds.t.isel(
                time=slice(0, 2), level=0, latitude=slice(7, 14), longitude=slice(0, 7)
            ).values
        ),
    )


def test_era5_chunked_retrieval_correct_values_fixed_level_3d():
    ds = xr.open_dataset(temperature_path)
    era5 = Era5(
        temperature_path,
        "t",
        level=1,
        normalize=False,
        chunker=ContinuousChunker((2, 7, 7)),
    )

    assert torch.allclose(
        era5[0],
        torch.Tensor(
            ds.t.isel(
                time=slice(0, 2), level=2, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[1],
        torch.Tensor(
            ds.t.isel(
                time=slice(2, 4), level=2, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[2],
        torch.Tensor(
            ds.t.isel(
                time=slice(4, 6), level=2, latitude=slice(0, 7), longitude=slice(0, 7)
            ).values
        ),
    )
    assert torch.allclose(
        era5[4],
        torch.Tensor(
            ds.t.isel(
                time=slice(0, 2), level=2, latitude=slice(7, 14), longitude=slice(0, 7)
            ).values
        ),
    )


def test_load_seamap():
    ds = xr.open_dataset(lsm_path)
    era5 = Era5(lsm_path, "lsm", normalize=False)

    assert era5[0].squeeze().shape == (ds.latitude.size, ds.longitude.size)
    assert torch.allclose(era5[0].squeeze(), torch.Tensor(ds.lsm.isel(time=0).values))
    assert torch.allclose(era5[1].squeeze(), torch.Tensor(ds.lsm.isel(time=1).values))
    assert torch.allclose(era5[3].squeeze(), torch.Tensor(ds.lsm.isel(time=2).values))


# fixtures
@pytest.fixture(scope="session")
def mock_dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "mock_dataset.nc"
    a = torch.Tensor([[1, 2, 3], [4, 5, 6]]).unsqueeze(0).unsqueeze(0)
    latitude = np.array([10, 20])
    longitude = np.array([30, 40, 50])
    time = np.array([-1])
    level = np.array([-10])
    ds = xr.Dataset(
        data_vars={"t": (("level", "time", "latitude", "longitude"), a)},
        coords={
            "level": level,
            "longitude": longitude,
            "latitude": latitude,
            "time": time,
        },
    )
    ds.to_netcdf(str(path))
    return path


# fixtures
@pytest.fixture(scope="session")
def mock_dataset_euclidean(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "mock_dataset_euclidean.nc"
    a = torch.Tensor([[1, 2, 3], [4, 5, 6]]).unsqueeze(0).unsqueeze(0)
    latitude = np.array([90, 0])
    longitude = np.array([0, 90, 180])
    time = np.array([-1])
    level = np.array([-10])
    ds = xr.Dataset(
        data_vars={"t": (("level", "time", "latitude", "longitude"), a)},
        coords={
            "level": level,
            "longitude": longitude,
            "latitude": latitude,
            "time": time,
        },
    )
    ds.to_netcdf(str(path))
    return path


# fixtures
@pytest.fixture(scope="session")
def mock_dataset_lengths(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "mock_dataset_euclidean.nc"
    a = np.random.random((1, 21, 60, 90))
    latitude = np.arange(60)
    longitude = np.arange(90)
    time = np.arange(21)
    level = np.array([-10])
    ds = xr.Dataset(
        data_vars={"t": (("level", "time", "latitude", "longitude"), a)},
        coords={
            "level": level,
            "longitude": longitude,
            "latitude": latitude,
            "time": time,
        },
    )
    ds.to_netcdf(str(path))
    return path


def test_dataset_coords_sphere(mock_dataset):
    ds = xr.open_dataset(mock_dataset)
    era5 = Era5(
        ds, "t", normalize=False, coords=CoordinateBundle.SPHERE, normalize_coords=False
    )

    assert era5[0].shape == (5, 2, 3)
    assert torch.allclose(
        era5[0],
        torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[-10, -10, -10], [-10, -10, -10]],
                [[-1, -1, -1], [-1, -1, -1]],
                [[10, 10, 10], [20, 20, 20]],
                [[30, 40, 50], [30, 40, 50]],
            ]
        ),
    )


def test_dataset_coords_euclidean(mock_dataset_euclidean):
    ds = xr.open_dataset(mock_dataset_euclidean)
    era5 = Era5(
        ds,
        "t",
        normalize=False,
        coords=CoordinateBundle.EUCLIDEAN,
        normalize_coords=False,
    )

    # just saving some space
    t0 = torch.Tensor([0])
    t1 = torch.Tensor([1])
    x00, y00, z00 = t0, t0, t1
    x01, y01, z01 = t0, t0, t1
    x02, y02, z02 = t0, t0, t1
    x10, y10, z10 = t1, t0, t0
    x11, y11, z11 = t0, t1, t0
    x12, y12, z12 = -t1, t0, t0
    x = [[x00, x01, x02], [x10, x11, x12]]
    y = [[y00, y01, y02], [y10, y11, y12]]
    z = [[z00, z01, z02], [z10, z11, z12]]
    assert torch.allclose(
        era5[0],
        torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[-10, -10, -10], [-10, -10, -10]],
                [[-1, -1, -1], [-1, -1, -1]],
                x,
                y,
                z,
            ]
        ),
        atol=1e-6,
    )


def test_dataset_correct_length_unchunked(mock_dataset_lengths):
    ds = xr.open_dataset(mock_dataset_lengths)
    era5 = Era5(ds, "t", normalize=False)

    assert len(era5) == 21


def test_dataset_correct_length_chunked_2d(mock_dataset_lengths):
    ds = xr.open_dataset(mock_dataset_lengths)
    era5 = Era5(ds, "t", normalize=False, chunker=ContinuousChunker((2, 2)))
    assert len(era5) == 21 * 30 * 45

    era5 = Era5(ds, "t", normalize=False, chunker=ContinuousChunker((10, 10)))
    assert len(era5) == 21 * 6 * 9


def test_dataset_correct_length_chunked_3d(mock_dataset_lengths):
    ds = xr.open_dataset(mock_dataset_lengths)
    era5 = Era5(ds, "t", normalize=False, chunker=ContinuousChunker((3, 2, 2)))
    assert len(era5) == 7 * 30 * 45

    era5 = Era5(ds, "t", normalize=False, chunker=ContinuousChunker((7, 10, 10)))
    assert len(era5) == 3 * 6 * 9


def test_dataset_coordinates_specifying_sphere(mock_dataset):
    ds = xr.open_dataset(mock_dataset)
    era5 = Era5(
        ds,
        "t",
        normalize=False,
        coords=[Coordinate.LATITUDE, Coordinate.LONGITUDE],
        normalize_coords=False,
    )

    assert era5[0].shape == (3, 2, 3)
    assert torch.allclose(
        era5[0],
        torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[10, 10, 10], [20, 20, 20]],
                [[30, 40, 50], [30, 40, 50]],
            ]
        ),
    )

    era5 = Era5(
        ds,
        "t",
        normalize=False,
        coords=[Coordinate.TIME, Coordinate.LEVEL],
        normalize_coords=False,
    )

    assert era5[0].shape == (3, 2, 3)
    assert torch.allclose(
        era5[0],
        torch.Tensor(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[-1, -1, -1], [-1, -1, -1]],
                [[-10, -10, -10], [-10, -10, -10]],
            ]
        ),
    )


def test_dataset_coordinates_specifying(mock_dataset_euclidean):
    ds = xr.open_dataset(mock_dataset_euclidean)
    era5 = Era5(
        ds,
        "t",
        coords=[Coordinate.Y, Coordinate.X],
        normalize=False,
        normalize_coords=False,
    )
    t0 = torch.Tensor([0])
    t1 = torch.Tensor([1])
    x00, y00 = t0, t0
    x01, y01 = t0, t0
    x02, y02 = t0, t0
    x10, y10 = t1, t0
    x11, y11 = t0, t1
    x12, y12 = -t1, t0
    x = [[x00, x01, x02], [x10, x11, x12]]
    y = [[y00, y01, y02], [y10, y11, y12]]
    assert torch.allclose(
        era5[0],
        torch.Tensor([[[1, 2, 3], [4, 5, 6]], y, x]),
        atol=1e-6,
    )
