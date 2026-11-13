"""
Controlled random states for sequential CDF-RAG experiments.

Scope:
- Python random module
- NumPy legacy global RNG
- PyTorch global RNG

Does NOT control Ollama sampling, Python hash
randomization, or separately constructed RNG objects.
"""

import random


MAX_SEED = 2**32 - 1


def validate_seed(seed):
    if (
        type(seed) is not int
        or not 0 <= seed <= MAX_SEED
    ):
        raise ValueError(
            f"Seed must be an integer from "
            f"0 to {MAX_SEED}"
        )

    return seed


def seed_libraries(seed):
    validate_seed(seed)

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return {
        "seed": seed,
        "python_random": True,
        "numpy_legacy_global_rng": True,
        "torch_global_rng": True,
        "ollama_seed_enforced": False,
        "python_hash_seed_enforced": False,
        "torch_deterministic_algorithms_enforced": False,
        "independent_rng_objects_controlled": False,
    }
