#!/bin/bash
cd $REPO
until grep -q -E "MODEL_DONE|Traceback" $SCRATCH/chain_model.log; do sleep 30; done
FRAMES=ESP_045994_1985,ESP_046060_1985,ESP_068294_1985,ESP_068360_1985,ESP_068426_1985,ESP_068822_1985,ESP_073068_1985,PSP_002387_1985 PROCS=4   $SCRATCH/venv/bin/python -I -W ignore scripts/run_anywhere.py 18.44715 77.41266 128 $SCRATCH/hirise_rdr $SCRATCH/recognizer_thr0.85 $SCRATCH/hero/restored_v2 $SCRATCH/hero/rover_truth_traverse $SCRATCH/anywhere_site1
echo ANYWHERE_DONE
