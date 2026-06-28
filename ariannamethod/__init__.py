"""
ariannamethod — the notorch shim that gives PostGPT-Q a from-scratch, torch-free trainer.

C line:  notorch.c / notorch.h  (pure-C neural framework — tape autograd, RRPRAM,
                                 Janus-echo primitives, Chuck; q's lineage)
Python:  notorch_nn.py          (ctypes; QEngine builds q's triple-gated transformer)
         chuck.py               (the self-aware optimizer, nt_tape_chuck_step)

q's weights were originally trained on an A100 with PyTorch. This trains q's ε
substrate with zero torch and exports the same QPTQ .bin the inference engines load.
"""

from .notorch_nn import Tensor, QEngine, softmax, multinomial, seed
from .chuck import ChuckOptimizer

__all__ = ['Tensor', 'QEngine', 'softmax', 'multinomial', 'seed', 'ChuckOptimizer']
