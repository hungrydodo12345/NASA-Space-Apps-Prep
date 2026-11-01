#!/bin/bash
set -x
cd $REPO
rm -rf $SCRATCH/hero/rover_truth_traverse $SCRATCH/hero/restored_v2
PROCS=4 scripts/build_truth_traverse.sh $SCRATCH $SCRATCH/venv/bin/python 121 600 30 $SCRATCH/hero/rover_truth_traverse
PROCS=4 $SCRATCH/venv/bin/python -I -W ignore scripts/build_rover_truth_v3.py $SCRATCH/navcam $SCRATCH/navcam_right $SCRATCH/rover/meta $SCRATCH/rover/meta_right $SCRATCH/ai4mars/ai4mars-dataset-merged-0.6/m2020/labels $SCRATCH/rover/waypoints.json $SCRATCH/rover/hero_navcam.json $SCRATCH/rover/stereo_pairs.json $SCRATCH/hirise_jezero/DTEEC_045994_1985_046060_1985_U01.IMG $SCRATCH/hero/rover_truth_traverse
echo TRUTH_DONE
PROCS=4 $SCRATCH/venv/bin/python -I -W ignore scripts/restore_real_v2.py $SCRATCH/hirise_jezero $SCRATCH/hirise_rdr $SCRATCH/hero/rover_truth_traverse $SCRATCH/hero/restored_v2
echo RESTORE_DONE
for t in 0.85 0; do ALIGN=0 THR=$t $SCRATCH/venv/bin/python -I -W ignore scripts/rocks_real_matched.py $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse data_products/texture_filter.json results/rocks_traverse_thr$t.json; done
echo CHAIN_DONE
