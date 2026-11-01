#!/bin/bash
S=$SCRATCH
cd $REPO
until grep -q -E "NN_FIG_DONE|Traceback" $S/chain_nn.log; do sleep 30; done
FRAMES=ESP_045994_1985,ESP_046060_1985,ESP_068294_1985,ESP_068360_1985,ESP_068426_1985,ESP_068822_1985,ESP_073068_1985,PSP_002387_1985 PROCS=3 \
  $S/venv/bin/python -I -W ignore scripts/run_anywhere.py 18.44715 77.41266 128 $S/hirise_rdr $S/recognizer_thr0 $S/hero/restored_v2 $S/hero/rover_truth_traverse $S/anywhere_site1
echo ANYWHERE_DONE
