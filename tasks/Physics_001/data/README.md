# Coordinate coverage of the simulated arrays

`MATBG Superfluid Stiffness Core Dataset.txt` is a text export of simulated arrays. The named coordinate and response arrays have the following lengths:

| Block | Coordinate length | Response lengths |
|---|---|---|
| Carrier density | `n_eff`: 50 | `D_s_conv`, `D_s_geom`, `D_s_exp_hole`, `D_s_exp_electron`: 50 each |
| Temperature | `T`: 100 | `D_s_bcs`: 95; `D_s_nodal`: 95; `D_s_power_n2`: 95; `D_s_power_n2_5`: 90; `D_s_power_n3`: 90; `D_s_experimental`: 110 |
| DC current | `I_dc`: 50 | `D_s_gl`: 60; `D_s_linear`: 50; `D_s_dc_exp`: 85 |
| Microwave | `P_mw`, `I_mw_amplitude`: 50 each | `D_s_mw_exp`: 50 |

No separate coordinate arrays or observation IDs in this export specify how to pair the unequal temperature and DC-current arrays. A corrected export or an explicit source coordinate mapping is needed before fitting those curves. Truncating, padding, or assigning a new grid would add a pairing assumption. The arrays labelled `D_s_bcs` and `D_s_power_n2` also contain identical values; their intended model definitions need the source generator or documentation.

## Experimental source data

The source article is [*Superfluid stiffness of magic-angle twisted bilayer graphene*](https://doi.org/10.1038/s41586-024-08494-7). Its Data availability statement describes access to the experimental data through a request to the corresponding authors, subject to the stated sponsor conditions. The simulated export in this directory is distinct from those original experimental measurements.

The numerical arrays are preserved as supplied. Their missing coordinate mappings and generator provenance remain to be established.
