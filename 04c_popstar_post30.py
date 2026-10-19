"""
POPSTAR post-30-minute decider
==============================
Identical to the post-test decider (04b): 100 best-RL-phase + 100 trough trials,
randomised and interleaved. Trial order is saved to
data/<subject>_post_30min_trial_order.csv.

By default this block gets its OWN freshly randomised order (different seed from 04b).
Set REUSE_ORDER_FROM = "post_test" to replay exactly the same order as the first post-test.
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
    BLOCK_LABEL = "post_30min"
    BLOCK_NAME = "Post-30-min test"
    BLOCK_INDEX = 3
    REUSE_ORDER_FROM = None   # or "post_test"
