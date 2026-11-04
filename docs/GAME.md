# Jezero Explorer: the game shell (branch `game-tiling`)

A browser game (three.js, no build step) over the real terrain of Jezero crater: open-world exploration of the whole landing region, plus a story mode that follows Perseverance's first months. Code: `web/game/`. Data: `scripts/build_game_tiles.py` (terrain) and `scripts/build_story.py` (story).

## The world

- **Area.** The full footprint of the HiRISE stereo DTM `DTEEC_045994_1985_046060_1985`, about 14.4 × 6.7 km of crater floor around the landing site. The root tile is a 14.4 km square; tiles without DTM data are not built.
- **Levels of detail.**
  - Everywhere: level 6 (225 m tiles, 0.9 m texels, 3.5 m mesh spacing).
  - Along the rover's real traverse (±80 m, 261 waypoints, sols 13–718) and within 300 m of the landing spot: level 8 (56 m tiles, 22 cm texels, the orthoimage's own resolution).
  - Over the 6 cm restoration and the Ingenuity mosaic: level 10 (14 m tiles, 5.5 cm texels).
- **Size.** 3,709 tiles with JPEG textures, about 0.8 GB. Coarse levels use 33 × 33 meshes, the rest 65 × 65.
- **Streaming.** The game loads a tile's children when the camera is closer than 240 × the tile's geometric error. It keeps at most 900 tiles in memory and evicts the least recently shown first.

## Modes

- **Open world.** Start at the landing spot and drive anywhere. Markers show the story locations (orange), the parachute and the bright boulder (white), and the rover's position every 12th waypoint (blue). The HUD shows the nearest point of interest, your position and the distance driven.
- **Story mode.** Eight chapters follow the mission. An orange beacon and an arrow lead to each objective; arriving opens the next chapter. Progress is saved in the browser.

| # | Chapter | Where (from data) |
|---|---|---|
| 1 | Touchdown | landing spot (site 3, drive 0) |
| 2 | First drive | rover position on sol 14 |
| 3 | Helicopter drop-off | rover position on sol 44 |
| 4 | The helicopter's view | Ingenuity truth site 2 (flights 4–5 mosaic) |
| 5 | Onto the crater floor | rover position on sol 110 |
| 6 | An empty tube | rover position on sol 164 (Roubion coring attempt) |
| 7 | First rock samples | rover position on sol 196 (Montdenier) |
| 8 | The debris field | the backshell, as placed in HiRISE by this project |

Rover positions are the last localized waypoint on or before each sol. Waypoints are in the rover team's map frame, and are moved onto the DTM by the landing-spot offset (`traverse.json`). That offset is exact at the landing spot; elsewhere, local differences between the two frames of about 10–20 m are possible. Chapter texts state only well-documented mission facts.

## Controls

<kbd>W</kbd>/<kbd>S</kbd> drive, <kbd>A</kbd>/<kbd>D</kbd> steer, <kbd>Shift</kbd> fast (25 m/s; normal 5 m/s, while the real rover's top speed is about 0.04 m/s), <kbd>C</kbd> camera (chase, 2 m mast camera, high view), <kbd>P</kbd> provenance view (red = measured texture, green = simulated detail), <kbd>M</kbd> menu, <kbd>Enter</kbd> continue the story.

## Run

```
python scripts/build_game_tiles.py <world> --region-dtm --base-level 6 --focus-m 600 --focus-level 8 --overlay-level 10 \
  --traverse waypoints.json --corridor-m 80 --corridor-level 8 --jpeg 85 --coarse-grid 33 --coarse-below 4 --workers 4 \
  --dtm ... --ortho ... --ortho-lbl ... --rdr-lbl ... --site-summary ... --center-global -4370690 -23476196 --chain ... --ingenuity ... --placement ...
python scripts/build_story.py <world> ESP_068294_1985_RED.LBL
ln -s <world> web/game/data && cd web/game && python -m http.server     # open http://localhost:8000
```

## Hooks for what comes next

- **Weather and lighting.** `Env` in `game.js` holds the sun azimuth and elevation, sky colour and dust level. `Env.apply()` updates the lights and fog; a day cycle, dust storms or seasons just animate those values. Note that the base texture has the 2016 afternoon shading baked in, so lighting far from that sun direction will look inconsistent. A shading-free albedo layer would be the next texture to add.
- **Rover driving.** `Rover.update(dt)` is a kinematic placeholder that follows the terrain using height samples under and ahead of the rover. Suspension, wheel–terrain contact and slip can replace it without touching the rest of the game.
- **Terrain queries.** `Terrain.heightAt(x, z)` gives the terrain height from the loaded tiles, for physics, placing objects and line-of-sight checks.
- **Content.** Chapters and points of interest live in `story.json`, which `build_story.py` regenerates; new chapters only need a location and text.
