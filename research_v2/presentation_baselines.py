"""Compatibility import for the original baseline module path.

Canonical B1-B6 implementation: research_v2.pipelines.baselines.
"""
from .pipelines.baselines import *  # noqa: F401,F403
from .pipelines.baselines import main

if __name__ == "__main__":
    main()
