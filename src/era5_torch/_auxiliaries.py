from typing import Optional
import torch
import xarray
from enum import Enum
import math
import numpy as np


class Coordinate(Enum):
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    LEVEL = "level"
    TIME = "time"
    X = "x"
    Y = "y"
    Z = "z"
    NULL = "null"


class CoordinateBundle(Enum):
    SPHERE = [
        Coordinate.LEVEL,
        Coordinate.TIME,
        Coordinate.LATITUDE,
        Coordinate.LONGITUDE,
    ]
    EUCLIDEAN = [
        Coordinate.LEVEL,
        Coordinate.TIME,
        Coordinate.X,
        Coordinate.Y,
        Coordinate.Z,
    ]


class CoordinateBundleTyper(Enum):
    """Typer enum for coordinate bundles, as the list value of the original enum raises an error."""

    SPHERE = "sphere"
    EUCLIDEAN = "euclidean"

    def to_enum(self):
        return CoordinateBundle[self.value.upper()]


def remove_auxiliary(x: torch.Tensor) -> torch.Tensor:
    """Default implementation to remove the auxiliary information from the tensor. Assumes that the auxiliary information is
    stored in additional channels along the first dimension.

    Args:
        x: The tensor with added auxiliary information. Auxiliary information must have been added from the same class.

    Returns:
        The tensor without auxiliary information.
    """
    return x[0]


def add_coords_information(
    sample: xarray.DataArray,
    coords: list[Coordinate],
    normalize: bool = True,
) -> torch.Tensor:
    """Transforms sample to a tensor with auxiliary information added along the first dimension.
    The sample must have only a single level and timestamp.
    The input tensor must have shape (1,x,y) or (x,y), the output shape will be (n,x,y), where n = len(coords).
    The spatial information will be added as a meshgrid in the first dimension, time and level time will be added flat to all inputs.
    For example for coordinates [level, time, latitude, longitude] the output a will be (4,x,y), where the first dimension contains the level, time, latitude and longitude information.
    a[1,x,y] will give the level of the sample, a[2, x, y] will give the time of the sample, a[3, x, y] will give the latitude at coordinates (x,y) and a[4, x, y] will give the
    the longitude at coordinates (x,y).

    In the future, we might implement this function for samples that run over multiple time stamps.

    Args:
        sample: The sample to add auxiliary information to.
        normalize: Whether to normalize the coordinates to be in the range [0, 1]. This should always be on, except if you have a very valid reason to do so (i.e. debugging),
        as the size of some coordinates (particularly time) is very large and can thus disrupt the gradients when training the model.
        coords: The coordinates to add to the sample. The order of the coordinates is important, as this represents the order of the auxiliary information in the output tensor.

    Returns:
        A tensor representation of the sample with the auxiliary information added in the first dimension of shape (n,width,height). The coordinates will be added in the order that coords specifies.
    """
    a = torch.Tensor(sample.values)
    if len(a.shape) == 2:
        a = a.unsqueeze(0)
    if len(coords) == 0:
        # shortcut for no added coordinates
        return a

    latitude = torch.Tensor(sample.latitude.values)
    longitude = torch.Tensor(sample.longitude.values)
    time = sample.time.values
    level = sample.level.values

    if time.size != 1:
        raise ValueError(
            f"Only a single time stamp is supported at the moment, input has {time.size}"
        )
    if level.size != 1:
        raise ValueError(
            f"The sample must have only a single level, input has {time.size}"
        )

    if normalize:
        level_value = (level / 1000).item()
        time_value = (
            time.item() / np.datetime64("2009-01-01T00:00:00.000000000", "ns").item()
        )
    else:
        level_value = level.item()
        time_value = time.item()
    latitudes_view = torch.broadcast_to(torch.Tensor(latitude).view(1, -1, 1), a.shape)
    longitudes_view = torch.broadcast_to(
        torch.Tensor(longitude).view(1, 1, -1), a.shape
    )
    times_view = torch.full_like(a, time_value)
    level_view = torch.full_like(a, level_value)

    possible_coords = {"level": level_view, "time": times_view}
    # making lat-lon grid
    if normalize:
        possible_coords["latitude"] = latitudes_view / 180 + 1 / 2
        possible_coords["longitude"] = longitudes_view / 360 + 1 / 2
    else:
        possible_coords["latitude"] = latitudes_view
        possible_coords["longitude"] = longitudes_view

    # debug coordinates
    possible_coords["null"] = torch.full_like(a, 0)

    # making x-y-z grid
    x, y, z = transform_polar_to_euclidean(latitudes_view, longitudes_view)
    if normalize:
        x = x / 2 + 1 / 2
        y = y / 2 + 1 / 2
        z = z / 2 + 1 / 2
    possible_coords["x"] = x
    possible_coords["y"] = y
    possible_coords["z"] = z

    auxiliary_information = []
    for coord in coords:
        # coord.value just gives us the name of the coord, as coord is an enum
        coord_value = possible_coords.get(coord.value)
        if coord_value is None:
            # this should never happen to a user, as we use an enum
            raise ValueError(f"INTERNAL ERROR: Coordinate {coord} is not supported.")
        else:
            auxiliary_information.append(possible_coords[coord.value])

    return torch.cat([a] + auxiliary_information, dim=0)


def add_land_sea_mask(x: xarray.DataArray, lsm_dataset: xarray.Dataset) -> torch.Tensor:
    """
    Class for a callable sample transform, that adds a land-sea mask to the sample in the first dimension.

    Args:
        lsm_path: The path to the land-sea mask file. The file must be in netcdf format and must contain a variable
        named lsm, that contains the land-sea mask. The mask must contain the exact same coordinates as the origin data.
    """
    if "time" in x.coords:
        lsm = lsm_dataset.sel(
            time=x.time,
            latitude=x.latitude,
            longitude=x.longitude,
            method="nearest",
        )
    else:
        lsm = lsm_dataset.sel(
            latitude=x.latitude, longitude=x.longitude, method="nearest"
        )
    lsm_tensor = torch.Tensor(lsm["lsm"].values).unsqueeze(0)
    x_tensor = torch.Tensor(x.values)
    if len(x_tensor.shape) < len(lsm_tensor.shape):
        x_tensor = x_tensor.unsqueeze(0)
    if len(x_tensor.shape) != len(lsm_tensor.shape):
        raise ValueError("The shape of the sample and the land-sea mask do not match.")
    return torch.cat([x_tensor, lsm_tensor], 0)


def transform_polar_to_euclidean(
    lat: torch.Tensor, lng: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transforms polar coordinates (latitude, longitude) to euclidean coordinates.
    Both sets of coordinate arrays must have the exact same shape. Each pair of coordinates
    (lat[i],long[i]) produces one triple (x[i],y[i],z[i])

    We assume that the spherical coordinates follow standard latitude/longitude conventions, e.g.
    the latitude is 0 on the equator and 90 on the northpole.

    Args:
        lat: Tensor containing the latitude in degrees, values range from [-90,90]
        longitude: Tensor containing the longitude in degrees, values range from [-180,180]
    """
    # the radius of earth is set to 1 wlog
    if lat.shape != lng.shape:
        raise ValueError(f"Shapes don't match: {lat.shape} != {lng.shape}")
    r = 1
    # conversion to radians
    # transform latitude: physics convention is that theta is the inclination angle from the north pole
    # towards the equator
    lat_r = (-lat + 90) / 360 * 2 * math.pi
    lng_r = lng / 360 * 2 * math.pi
    # ---
    x = r * torch.cos(lng_r) * torch.sin(lat_r)
    y = r * torch.sin(lng_r) * torch.sin(lat_r)
    z = r * torch.cos(lat_r)
    return x, y, z
