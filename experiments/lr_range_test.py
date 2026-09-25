#!/usr/bin/env python
r"""Learning-rate range test for train_partial_parallel.py.

Runs train_partial_parallel unchanged, only overriding CONFIG: the whole run is
a linear warmup from 1e-6 to PEAK, so the LR sweeps the full range in one go.

In W&B, plot loss_train against learning_rate. Ignore the first ~10 epochs (the
model leaving its initialisation). LR_max is where loss_train stops decreasing
or turns NaN; use peak_learning_rate = LR_max / 3 for the real run.

Run from the repository root (~10 min on 4 GPUs):

    python -m experiments.lr_range_test
    LR_RANGE_TEST_PEAK=3e-3 python -m experiments.lr_range_test
"""
import os

import wandb

from experiments import train_partial_parallel as tpp

PEAK = float(os.environ.get("LR_RANGE_TEST_PEAK", 5e-4))

OVERRIDES = {
    # the run is only the warmup: one linear ramp from init to peak
    "epochs": 60,
    "warmup_epochs": 60,
    "init_learning_rate": 1e-6,
    "peak_learning_rate": PEAK,
    # optax needs decay_steps > warmup_steps; the cosine never starts
    "lr_decay_epochs": 100,
    "end_learning_rate": PEAK,
    # no dropout: it would blur the point where the LR starts to hurt
    "dropout": None,
    # residual plots at ~1e-4 / 3e-4 / 5e-4 as a visual check
    "log_freq": 20,
    "wandb_project": "Test-AE-partial-3-parallel-lr-range",
    "wandb_name": f"range-test-peak-{PEAK:.0e}",
}


if __name__ == "__main__":
    tpp.CONFIG.update(OVERRIDES)
    tpp.train(runid=wandb.util.generate_id())
