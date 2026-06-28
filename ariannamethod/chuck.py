"""
chuck.py — the Chuck optimizer via notorch (nt_tape_chuck_step). No torch.optim.

The self-aware optimizer (loss-aware damping, per-parameter gradient monitoring,
stagnation noise, parameter freezing, macro-patience) — the real C implementation,
used here to train PostGPT-Q's ε substrate from scratch without PyTorch.
"""

from .notorch_nn import _lib
import ctypes


class ChuckOptimizer:
    def __init__(self, lr=3e-4, max_grad_norm=1.0):
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def step(self, loss_val):
        self.global_step += 1
        _lib.nt_tape_clip_grads(ctypes.c_float(self.max_grad_norm))
        _lib.nt_tape_chuck_step(ctypes.c_float(self.lr), ctypes.c_float(loss_val))

    def zero_grad(self):
        pass  # notorch tape is reset by nt_tape_clear()
