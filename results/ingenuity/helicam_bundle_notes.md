# Where the helicopter's location metadata lives

- NASA raw-image feed (what `scripts/ingenuity/fetch_flight*.py` use): no position. `xyz` and `attitude` are `UNK`; only `sclk`, image size and a scale factor are filled.
- **PDS Imaging Node bundle `mars2020_helicam`**: https://pds-imaging.jpl.nasa.gov/data/mars2020/mars2020_helicam/
  - `data/sol/<sol, 5 digits>/ids/{edr,fdr,rdr}/heli/` per-frame products; RDR labels (`*RAS*.xml`, `*RAF*`, `*RAD*`, `*RZS*`) contain the camera model in `HELI_G_FRAME_<flight>`, that frame's origin offset and quaternion relative to `SITE_FRAME <n>`, and a per-image `HELI_M_FRAME` pose.
  - `data/sol/<sol>/ids/rdr/mosaic/` JPL-made mosaics, e.g. `V_MMMM_0414_RAS_0030000_VRT_S_F26LS01J01.IMG` (flight 26, vertical).
  - `data/collection_data_inventory.csv` lists 34,986 products; the mosaic types include `vrt` (vertical), `orr` (orthorectified) and `dem`.
- Rover site frames are separate: the rover's MMGIS waypoint file lists sites and positions (`site`, `easting`, `northing`, `lat`, `lon`).
