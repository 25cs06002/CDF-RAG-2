"""Compatibility import; canonical B7 adapter lives in pipelines/.

Historical imports continue to resolve to the same Python class.
"""
from .pipelines.legacy_no_rl import *  # noqa: F401,F403
