# Orchestration scripts

These are the exact shell chains used to produce the traverse-wide results, with machine paths replaced by
`$SCRATCH` (the working data folder) and `$REPO` (this repository).

| Script | What it runs |
|---|---|
| `chain_all.sh` | Builds rover truth v3 along the traverse (sols 121–600) and for the original stops, then the 8-pass restorations (`scripts/restore_real_v2.py`), then the pre-registered blob-detector rock test. |
| `chain_model.sh` | Trains and scores the Stage-1 recogniser with spatial cross-validation (`scripts/train_recognizer.py`) and renders the grounded comparison (`scripts/render_grounded.py`). |
| `chain_anywhere.sh` | Runs the full model at a site with no rover imagery (`scripts/run_anywhere.py`). |
| `hirise_coverage_all.py` | Counts the usable HiRISE passes covering every rover stop. Its output is `data_products/derived/hirise_coverage_all.json`. |
| `chain_nn.sh` | Current order of the model stage: trains and scores the recogniser first, then renders the figure, then the UCL-style restorations. It replaces `chain_model.sh`. |
| `chain_wide.sh` | Amendment 1: builds the ±16 m rover truth, then runs the pre-registered rock test on it. |
