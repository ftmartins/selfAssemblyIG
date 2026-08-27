"""
Small filesystem / serialization helpers shared across the package.
"""

import os

import numpy as np


def save_pickle_npy(path: str, obj) -> None:
    """``np.save`` an arbitrary Python/JAX object (pytree) with pickling on."""
    np.save(path, obj, allow_pickle=True)


def load_pickle_npy(path: str):
    """Inverse of :func:`save_pickle_npy`.  Returns ``None`` if ``path`` is missing."""
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True).item()


def rename_with_backup(filepath: str) -> str:
    """
    If ``filepath`` exists, move it aside to ``name_old.ext`` (or
    ``name_old_1.ext``, ``name_old_2.ext`` …) and return the original path so the
    caller can write a fresh file there.  Otherwise return ``filepath`` unchanged.
    """
    if not os.path.exists(filepath):
        return filepath
    directory, filename = os.path.split(filepath)
    name, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, f"{name}_old{ext}")
    counter = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{name}_old_{counter}{ext}")
        counter += 1
    os.rename(filepath, candidate)
    print(f"File renamed to: {candidate}")
    return filepath
