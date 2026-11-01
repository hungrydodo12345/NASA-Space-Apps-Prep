# Geolocation finding: rover waypoints vs our HiRISE orthoimage

Recorded 2026-11-01.

## What we found

The parked Perseverance rover is visible in three early HiRISE passes: ESP_068294, ESP_068360 and ESP_068426, taken on sols 1–11, before its first drive. We matched a 128 m window of ESP_068294 around the rover to the ESP_045994 orthoimage. The match is unique (score 0.55; the next best location scores 0.12).

The result: the rover's actual position in our orthoimage is **138 m north and 19 m east** of where the published waypoint 3_0, converted with `groundtruth.latlon_to_ortho`, places it.

## Is it the same everywhere?

A second check used ESP_068822 at sol 41, with the rover at stop 3_1392. After applying the same correction, a bright, rover-like object sits about 16 m from the predicted position. So the error is mostly a fixed offset of about 140 m, plus a part that varies across the site and is on the order of 10–20 m.

## Consequences

- **All rock-level comparisons are invalid.** Every rover-stop HiRISE window used in them (`restore_real_v2.py` and the rock tests) was centred about 140 m from the rover's true position. HiRISE was therefore compared against rover imagery of different ground. This explains:
  - near-chance rock matching for every method;
  - the absence of any correlation peak between rover truth and HiRISE within ±16 m or ±60 m;
  - the failed registration attempts.
- **Unaffected results:**
  - the simulations;
  - the split-half (FRC) resolution measurements, which compare HiRISE with HiRISE;
  - the known-object rover test, which locates the rover in HiRISE itself.

## Still needed

Per-stop placement of the rover truth on HiRISE to within about 25 cm. Two candidates:

1. Match the ESP_068822 and ESP_073068 passes against rover tracks or the rover at many stops.
2. Use JPL's official orthomosaics (`mars2020_navcam_ops_mosaic`, ORR products, downloaded for 13 stops) with a robust registration method.

The JPL DEM products tried so far have only about 11% coverage, which is too sparse for terrain matching.
