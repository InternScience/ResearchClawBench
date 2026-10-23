# Packaged FuXi sample

The two NetCDF files in this directory are samples on a **1-degree grid**. Their headers specify the following dimensions and coordinates:

| File | `data` dimensions | Time coverage |
|---|---|---|
| `20231012-06_input_netcdf.nc` | `(time=2, level=70, lat=181, lon=360)` | 2023-10-12 00:00 and 06:00 UTC |
| `006.nc` | `(time=1, step=1, level=70, lat=181, lon=360)` | One initialization, one forecast step (`step=6`, described in the file as a 6-hour forecast) |

Both files have latitude coordinates from 90 to -90 and longitude coordinates from 0 to 359. Their `data.description` attributes identify them as 1-degree sample input/output. The output file contains a single forecast lead.

The full-resolution FuXi model uses inputs of shape `(2, 70, 721, 1440)` on a 0.25-degree grid. The packaged 1-degree arrays therefore have a different input shape from the upstream model. A full forecast experiment also needs the model files, the remaining forecast leads, and verifying/reference data appropriate to that experiment.

## Upstream resources

- [FuXi paper](https://doi.org/10.1038/s41612-023-00512-1).
- [Author repository: input preparation and channel order](https://github.com/tpys/FuXi/blob/cbfc51136d3f477797d7046ad3410a729e5d244a/README.md#data-preparation).
- [Author repository: model and sample-data distribution](https://github.com/tpys/FuXi/blob/cbfc51136d3f477797d7046ad3410a729e5d244a/README.md#installation), including [Zenodo record 10401602](https://doi.org/10.5281/zenodo.10401602). The upstream README also describes its sample output as T+6 only.
- [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) for ERA5, as referenced in the paper's Data availability statement.
