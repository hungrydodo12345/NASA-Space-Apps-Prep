#!/bin/bash
cd $REPO
THR_CLEAN=0 ITERS=2500 $SCRATCH/venv/bin/python -I -W ignore scripts/train_recognizer.py $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse data_products/texture_filter.json $SCRATCH/recognizer_thr0
echo NN_TRAINED
STOPS=$(python3 -c "import json;d=json.load(open('$SCRATCH/recognizer_thr0/ucl_yardstick_thr0.0.json'))['per_stop'];print(','.join(r['stop'] for r in sorted(d,key=lambda r:-sum(r['truth'].values()))[:3]))")
echo "figure stops: $STOPS"
$SCRATCH/venv/bin/python -I -W ignore scripts/render_grounded.py $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse $SCRATCH/recognizer_thr0 $REPO/results/paper/fig_grounded_model.png $STOPS
echo NN_FIG_DONE
GPT_STYLE=1 GPT_ONLY=1 PROCS=4 $SCRATCH/venv/bin/python -I -W ignore scripts/restore_real_v2.py $SCRATCH/hirise_jezero $SCRATCH/hirise_rdr $SCRATCH/hero/rover_truth_traverse $SCRATCH/hero/gptstyle
echo GPT_DONE
