"""
POPSTAR post-test decider
=========================
200 trials: 100 at the best RL phase, 100 at the trough, randomised and interleaved.
Requires the RL decider (04a) to have been run first for this subject.
Trial order is saved to data/<subject>_post_test_trial_order.csv.
All logic lives in popstar_common.PostTestDecider.
"""
import os
import sys

try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    pass

from popstar_common import PostTestDecider


class Decider(PostTestDecider):
    BLOCK_LABEL = "post_test"
    BLOCK_NAME = "Post-test"
    BLOCK_INDEX = 2
