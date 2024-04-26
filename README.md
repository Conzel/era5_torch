# ERA5-Torch
This repository provides a wrapper around [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview) files that is compatible with PyTorch. 

## Installation
Install the requirements via `pip install -r requirements.txt`.

## Usage Examples
The main functionality is exposed through the Era5 class, which is a valid PyTorch Dataset and can thus be used with standard PyTorch dataloaders. 

Assume that you have downloaded some NetCDF files (for example, follow [this tutorial](https://github.com/joaohenry23/Download_ERA5_with_python) containing the temperature ("t") and your data lives at `path/to/dataset.nc`. You can then load the data with:

```
from era5_torch import Era5
ds = Era5("/path/to/dataset.nc", "t", normalize=False)
ds[0]
```

which returns a single sample of the temperature values with shape `(1, num_lat, num_long)`. By default, the samples are all located at a single point in time and a single pressure level. 

If your model uses coordinate information, you can pass in the coordinates desired coordinates to the dataset class, which are appended in the passed order to the first axis of the return sample:

```
from era5_torch import Era5, Coordinate
ds = Era5("/path/to/dataset.nc", "t", normalize=False, normalize_coords=True, coords=[Coordinate.X, Coordinate.Y, Coordinate.Z])
ds[0]
```

which returns a sample with shape `(4, num_lat, num_long)`, and `ds[0][0]` is the temperature, `ds[0][1]` the x-values of the temperature, et cetera.

To access the dataset in different ways (for example with 3D chunks of shape `(time, lat, long)`), we provide the Chunker class. This can be passed to the Era5 class.

```
from era5_torch import Era5, Coordinate, ContinuousChunker
ds = Era5("/Users/almico/datasets/era5_temperatures_2008.nc", "t", normalize=False,chunker=ContinuousChunker((3,3,3)))
ds[0]
```

Returns a sample of shape `(3,3,3)` with values varying over `(time, lat, long)`. Alternatively, to get patches, you can also create a Chunker with `ContinuousChunker((3,3))`. All chunkers have different behaviour, the `ContinuousChunker` for example returns chunks starting from the first position in the dataset (usually the (0,0) coordinate) and creates as many chunks in an orderly manner as possible by traversing the coordinates one after another. Refer to the documentation of the Chunkers to get more information.# era5_torch
