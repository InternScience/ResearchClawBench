# Meta data for GlaMBIE result datasets

## 1. Overview and dataset description

This dataset contains the GlaMBIE final combined results per region as well as the results for each data source individually.
The time resolution of the datasets is annual. Results exist for the hydrological year as well as for the calendar year.
Within GlaMBIE the solutions per data group are combined into one regional solution (Dataset 2.1) and then converted to calendar years for global aggregation (Dataset 2.2).

How to cite: GlaMBIE (2024): Glacier Mass Balance Intercomparison Exercise (GlaMBIE) Dataset 1.0.0. World Glacier Monitoring Service (WGMS), Zurich, Switzerland. https://doi.org/10.5904/wgms-glambie-2024-07

## 2. Dataset contents

### 2.1. Dataset in hydrological years
Within subfolder "/hydrological_years"

Contains all 19 regions

**Hydrological year columns:**
- start_dates [fractional years]: start dates in decimal years of the period of change
- end_dates [fractional years]: end dates in decimal years of the period of change
- glacier_area [km^2]: area in km^2 used for a specific time period
- combined_gt [Gt]: glacier change in Gigatonnes recorded by the combination of estimates between start and end dates
- combined_gt_errors [Gt]: glacier change error in Gigatonnes recorded by the combination of estimates between start and end dates
- combined_mwe [m w.e.]: glacier change in meter water equivalent recorded by the combination of estimates between start and end dates
- combined_mwe_errors [m w.e.]: glacier change error in meter water equivalent recorded by the combination of estimates between start and end dates
- <data_group>_gt [Gt]: glacier change in Gigatonnes recorded by a data group (altimetry, gravimetry or DEMifferencing+glaciological) between start and end dates
- <data_group>_gt_errors [Gt]: glacier change error in Gigatonnes recorded by a data group (altimetry, gravimetry or DEMdifferencing+glaciological) between start and end dates
- <data_group>_mwe [m w.e.]: glacier change in meter water equivalent recorded by a data group (altimetry, gravimetry or DEMdifferencing+glaciological) between start and end dates
- <data_group>_mwe_errors [m w.e.]: glacier change error in meter water equivalent recorded by a data group (altimetry, gravimetry or DEMdifferencing+glaciological) between start and end dates
- <data_group>_annual_variability [binary]: binary value describing if a data group provided annual variability or used the annual variability from the combination of estimates from other data sources. The value "1" means that the data group provided annual variability between start and end dates


### 2.2. Dataset in calendar years

Within subfolder "/calendar_years"

Contains all 19 regions and a global time series

**Calendar year columns:**
- start_dates [fractional years]: start dates in decimal years of the period of change
- end_dates [fractional years]: end dates in decimal years of the period of change
- glacier_area [km^2]: area in km^2 used for a specific time period
- combined_gt [Gt]: glacier change in Gigatonnes recorded by consensus estimate between start and end dates
- combined_gt_errors [Gt]: glacier change error in Gigatonnes recorded by the combination of estimates between start and end dates
- combined_mwe [m w.e.]: glacier change in meter water equivalent recorded by the combination of estimates between start and end dates
- combined_mwe_errors [m w.e.]: glacier change error in meter water equivalent recorded by the combination of estimates between start and end dates



