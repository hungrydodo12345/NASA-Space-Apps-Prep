#!/bin/bash
# Build rover truth v3 along the traverse in sol batches: fetch Navcam stereo pairs, build, delete raw images.
# Usage: scripts/build_truth_traverse.sh <scratch_dir> <python> <sol_min> <sol_max> <batch_sols> <truth_out>
S=$1; PY=$2; A=$3; B=$4; STEP=$5; OUT=$6
R=$(cd "$(dirname "$0")/.." && pwd)
for ((s=A; s<=B; s+=STEP)); do
  e=$((s+STEP-1)); W=$S/traverse_batch_$s
  python3 $R/scripts/fetch_navcam_stops.py $S/rover/waypoints.json $W $s $e || continue
  PROCS=${PROCS:-4} $PY -I -W ignore $R/scripts/build_rover_truth_v3.py $W/left $W/right $W/meta $W/meta_right \
     $S/ai4mars/ai4mars-dataset-merged-0.6/m2020/labels $S/rover/waypoints.json $W/items.json $W/pairs.json \
     $S/hirise_jezero/DTEEC_045994_1985_046060_1985_U01.IMG $OUT
  rm -rf $W/left $W/right
  echo "batch $s-$e done"
done
echo ALL_DONE
