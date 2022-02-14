# Generation/Simulation Tools for Planar Honeycomb Code

This repository contains code for creating stim circuits describing planar honeycomb patches,
performing Monte Carlo sampling of logical error rates from those circuits,
and producing related plots (for a paper).

### Run Tests

The following assumes you've cloned the repo and your current working directory is the repo root.

```bash
# (in a fresh virtual environment at repo root)
pip install -r requirements.txt
pytest src
```

### Collect and Plot Data

The following assumes you've cloned the repo and your current working directory is the repo root.

```bash
# (in a fresh virtual environment at repo root)
mkdir out
pip install -r requirements.txt

# Collect data (this example takes ~1 minute).
PYTHONPATH=src python src/hcb/artifacts/collect_logical_error_rates.py \
    -case_error_rates 0.001 0.002 \
    -case_observables H V \
    -case_gate_sets SD6 \
    -case_decoders pymatching \
    -case_distances 3 5 \
    -max_shots 10_000 \
    -max_errors 100 \
    -merge_mode saturate \
    -threads 4 \
    -storage_location out/sample_data.csv

# (...Wait for data collection to finish...)

# Plot collected data.
PYTHONPATH=src python src/hcb/artifacts/make_threshold_plots.py \
    out/sample_data.csv
```
