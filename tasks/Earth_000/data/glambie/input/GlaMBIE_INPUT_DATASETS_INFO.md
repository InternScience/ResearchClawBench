# Meta data for GlaMBIE input datasets

## 1. Overview and dataset description

Input datasets submitted to GlaMBIE. 
This collection of datasets represents the original data submitted to GlaMBIE with edits to fit the GlaMBIE monthly grid and format.

How to cite: GlaMBIE (2024): Glacier Mass Balance Intercomparison Exercise (GlaMBIE) Dataset 1.0.0. World Glacier Monitoring Service (WGMS), Zurich, Switzerland. https://doi.org/10.5904/wgms-glambie-2024-07

## 2. Dataset contents

### 2.1. Folders and naming conventions

There is one subfolder for each of the 19 GTN-G regions, including all datasets submitted for the regions.

Each CSV file represents one solution submitted for a specific region

The file naming follows the convention:
{region_name}_{data_source}_{group_name_defined_at_submission}_{dataset_citation}.csv

### 2.2. Dataset columns

- start_dates [fractional years]: start dates in decimal years of the period of change
- end_dates [fractional years]: end dates in decimal years of the period of change
- changes [m or Gt]: glacier change recorded in defined unit
- errors [m or Gt]: uncertainties on glacier change recorded in defined unit
- unit: unit of "changes" and "errors", either "Gt" (Gigatonnes) or "m" (meters)
- author: dataset citation used for the dataset within the GlaMBIE publication




