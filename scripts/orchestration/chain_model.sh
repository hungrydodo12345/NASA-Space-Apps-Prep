#!/bin/bash
cd $REPO
until grep -q -E "CHAIN_DONE|Traceback" $SCRATCH/chain_all.log; do sleep 30; done
echo "== traverse rock test (pre-registered, blob detector)"; tail -14 $SCRATCH/chain_all.log
for t in 0.85 0; do THR_CLEAN=$t ITERS=2500 $SCRATCH/venv/bin/python -I -W ignore scripts/train_recognizer.py $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse data_products/texture_filter.json $SCRATCH/recognizer_thr$t; done
$SCRATCH/venv/bin/python -I -W ignore scripts/render_grounded.py $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse $SCRATCH/recognizer_thr0.85 $REPO/results/paper/fig_grounded_model.png
echo MODEL_DONE
