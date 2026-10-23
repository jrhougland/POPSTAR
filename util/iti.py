"""ITI sampling for predetermined trials.

a scaled Beta distribution on [ITI_MIN, ITI_MAX] plus a point mass at SPIKE_VALUE,
with the Beta distribution solved so the mixture mean is TARGET_MEAN.
"""
import numpy as np

ITI_MIN = 2.5
ITI_MAX = 5.5

TARGET_MEAN = 3.15       # s, mean of the whole mixture
SPIKE_VALUE = 5.5        # s, the point mass
SPIKE_WEIGHT = 0.02     # fraction of draws at the spike
CONCENTRATION = 5.0      # Beta concentration; below ~6.5 the mode sits at ITI_MIN

# Solve the Beta so the MIXTURE (Beta distribution + point mass) has mean TARGET_MEAN:
#   TARGET_MEAN = SPIKE_WEIGHT * SPIKE_VALUE + (1 - SPIKE_WEIGHT) * _BETA_MEAN
_BETA_MEAN = (TARGET_MEAN - SPIKE_WEIGHT * SPIKE_VALUE) / (1 - SPIKE_WEIGHT)
_P = (_BETA_MEAN - ITI_MIN) / (ITI_MAX - ITI_MIN)
_ALPHA = _P * CONCENTRATION
_BETA = (1 - _P) * CONCENTRATION

assert ITI_MIN < _BETA_MEAN < ITI_MAX, (
    f"TARGET_MEAN {TARGET_MEAN} implies a Beta mean of {_BETA_MEAN:.3f}, outside "
    f"[{ITI_MIN}, {ITI_MAX}] -- _BETA would go negative and rng.beta would throw."
)


def sample_iti(rng: np.random.Generator) -> float:
    """Draw one ITI in seconds. rng is the decider's seeded Generator."""
    if rng.random() < SPIKE_WEIGHT:
        return float(SPIKE_VALUE)
    return float(ITI_MIN + (ITI_MAX - ITI_MIN) * rng.beta(_ALPHA, _BETA))