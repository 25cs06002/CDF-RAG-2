"""Compatibility import; canonical B8 PPO adapter lives in pipelines/.

Historical imports continue to resolve to the same Python class.
"""
from .pipelines.legacy_rl import *  # noqa: F401,F403
