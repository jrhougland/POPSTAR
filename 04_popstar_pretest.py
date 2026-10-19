"""
POPSTAR pre-test decider
==============================
Targets all 8 phase bins (the same bins the RL agent chooses from), 40 trials each
(320 total), in one randomised order. Run before RL training.
The order is saved to data/<subject>_pre_test_trial_order.csv before trial 1.
All logic lives in popstar_common.PreTestDecider.
"""
import os
import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from popstar_common import PreTestDecider


class Decider(PreTestDecider):
    BLOCK_LABEL = "pre_test"
    BLOCK_NAME = "Pre-test"
    BLOCK_INDEX = 0
