#!/bin/bash
# Amendment 1: rover truth to +-16 m (Navcam mast elevation <= -5 deg) for every stop sols 13-600, then the
# pre-registered rock test on it (same detector, matching and thresholds). Resumes at sol 313 (13-312 already built).
S=$SCRATCH
cd /home/user/NASA-Space-Apps-Prep
SKIP_DONE=1 MAXEL=-5 HALF=16 RMAX=16 PROCS=4 scripts/build_truth_traverse.sh $S $S/venv/bin/python 313 600 30 $S/hero/rover_truth_wide
echo WIDE_TRUTH_DONE
for t in 0.85 0; do ALIGN=0 THR=$t $S/venv/bin/python -I -W ignore scripts/rocks_real_matched.py $S/hero/restored_v2 $S/hero/rover_truth_wide data_products/texture_filter.json results/rocks_wide_thr$t.json; done
echo WIDE_ROCKS_DONE
