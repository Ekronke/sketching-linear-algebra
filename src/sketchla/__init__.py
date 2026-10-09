"""Randomized sketching for least squares and low-rank approximation."""

from .sketches import CountSketch, GaussianSketch, SRHT, Sketch, make_sketch
from .algorithms import randomized_svd, sketch_and_solve
from ._backend import HAS_CPP_KERNELS

__all__ = [
    "Sketch",
    "GaussianSketch",
    "SRHT",
    "CountSketch",
    "make_sketch",
    "sketch_and_solve",
    "randomized_svd",
    "HAS_CPP_KERNELS",
]
