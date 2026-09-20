#!/usr/bin/env python
r"""Learning-rate range test for train_partial_parallel.py.

Why
---
In the 2000-epoch run (Student-2-parallel), loss_train and loss_test flatten
around epoch 250, which is exactly where the cosine schedule reaches
end_learning_rate = 2e-7. The plateau is the schedule dying, not convergence:
85% of the run happens at 1% of the peak LR. Before spending hours on a new
schedule, we need to know how high peak_learning_rate can actually go.

What it does
------------
It reuses train_partial_parallel unchanged and only overrides CONFIG, so the
model, the loss, the sharding and the data pipeline are bit-for-bit the ones of
the real run. The trick: optax's warmup_cosine_decay_schedule ramps *linearly*
from init_value to peak_value over warmup_steps, so a run that is nothing but
warmup sweeps the LR across the whole range of interest in one go.

    epochs = warmup_epochs = 60, init 1e-6 -> peak 5e-4

gives, reading the `learning_rate` axis W&B already logs every epoch:

    epoch   6 -> 5e-5      epoch  36 -> 3e-4
    epoch  12 -> 1e-4      epoch  48 -> 4e-4
    epoch  24 -> 2e-4      epoch  60 -> 5e-4

How to read the result
----------------------
In W&B, plot loss_train against learning_rate (not against step).

  - Epochs 1-10 drop steeply whatever the LR: that is the model leaving its
    initialisation, not a statement about the LR. Ignore that part.
  - Past that, the LR is too high from the epoch where loss_train stops
    decreasing and turns back up. Call that LR_max.
  - Take peak_learning_rate = LR_max / 3 for the real run.
  - If loss_train is still descending at 5e-4, nothing broke in the probed
    range: either take 2e-4 (safe, 10x the current peak) or re-run this probe
    with LR_RANGE_TEST_PEAK=3e-3 to find the real ceiling.

A divergence here is the expected outcome of a range test, not a failure: the
whole point is to make the 3-hour run discover nothing.

Cost: 60 epochs at ~10 s = ~10 min of compute, plus the dataset load and the
first-epoch XLA compilation (~1 min). Fits the Jean Zay dev QoS easily.

Run it from the repository root, exactly like the real training:

    python -m experiments.lr_range_test
"""
import os

import wandb

from experiments import train_partial_parallel as tpp

# Ceiling of the sweep. Override without editing the file:
#     LR_RANGE_TEST_PEAK=3e-3 python -m experiments.lr_range_test
PEAK = float(os.environ.get("LR_RANGE_TEST_PEAK", 5e-4))

OVERRIDES = {
    # the run IS the warmup: a single linear ramp from init to peak
    "epochs": 60,
    "warmup_epochs": 60,
    "init_learning_rate": 1e-6,
    "peak_learning_rate": PEAK,
    # optax needs decay_steps > warmup_steps (the cosine lasts
    # decay_steps - warmup_steps); the run stops before the cosine ever starts
    "lr_decay_epochs": 100,
    "end_learning_rate": PEAK,
    # no dropout: the 2000-epoch run has loss_test *below* loss_train, i.e. no
    # generalisation gap at all, so the regularisation only adds gradient noise
    # -- and here it would blur the point where the LR starts to hurt
    "dropout": None,
    # a residual plot + a checkpoint at epochs 20 / 40 / 60 (~1e-4 / 3e-4 /
    # 5e-4): the images degrade visibly once the LR is too high, which is a
    # useful second opinion on the scalar curve
    "log_freq": 20,
    # a separate project so the probe never sits next to the real runs
    "wandb_project": "Test-AE-partial-3-parallel-lr-range",
    "wandb_name": f"range-test-peak-{PEAK:.0e}",
}


if __name__ == "__main__":
    tpp.CONFIG.update(OVERRIDES)
    tpp.train(runid=wandb.util.generate_id())
