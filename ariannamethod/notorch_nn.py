"""
notorch_nn.py — PostGPT-Q (ε substrate) trainer API backed by libnotorch (ctypes).

NO PyTorch. Q's transformer substrate (θ = ε + γ + αδ — this trains ε) runs on
notorch (pure-C tensor/autograd) through ctypes. Q's weights were originally
trained on an A100 with torch; this gives Q a from-scratch, torch-free trainer
that exports the same QPTQ .bin the C/JS/Python inference engines load.

Q's triple attention, mapped to notorch tape ops (all verified):
  Content QK^T        -> nt_mh_causal_attention   (scale 1/sqrt(HD))
  RRPRAM x@Wr         -> nt_rrpram_attention       (scores = xn_i . Wr[h][:,p], causal)
  Janus echo          -> vjp ⊙ (wjp / ||wjp||) via nt_seq_rmsnorm + nt_scale + nt_mul
  per-mechanism gate  -> nt_sigmoid + nt_seq_gate  (out[t,d] = mech[t,d]*gate[t,gi])
  combine             -> nt_concat ; project -> nt_seq_linear (wo)
  MLP                 -> nt_seq_linear (up) -> nt_relu -> nt_seq_linear (dn)
  norm                -> nt_seq_rmsnorm with a FIXED ones gamma (q's norm is param-free)
  output (tied)       -> nt_seq_linear on tok ; loss -> nt_seq_cross_entropy

The per-mechanism gate bias gb IS trained, via the augmented-gate trick: gws is
[nm, D+1] and xn gets a constant-1 channel appended (nt_concat), so seq_linear gives
gws@xn + gb with gb as the last column — no broadcast-add op needed. The exporter splits
it back into gws[nm,D] + gbs[nm]. The magnitude "transformer gate" (clamp logit-mag) is
inference-only (applied by postgpt_q.c), so training optimizes the raw logits.
"""

import ctypes
import os
import math
import struct
import random
import platform

_dir = os.path.dirname(os.path.abspath(__file__))


def _find_or_build_lib():
    for ext in ('.dylib', '.so', '.dll'):
        p = os.path.join(_dir, f'libnotorch{ext}')
        if os.path.exists(p):
            return p
    src = os.path.join(_dir, 'notorch.c')
    if not os.path.exists(src):
        raise RuntimeError(f'notorch.c not found in {_dir}')
    is_darwin = platform.system() == 'Darwin'
    out = os.path.join(_dir, 'libnotorch.dylib' if is_darwin else 'libnotorch.so')
    import subprocess
    base = ['cc', '-O2', '-std=c11', '-shared', '-fPIC', '-o', out, src]
    attempts = []
    if is_darwin:
        attempts.append(base + ['-DUSE_BLAS', '-DACCELERATE', '-DACCELERATE_NEW_LAPACK',
                                '-framework', 'Accelerate', '-lm'])
    else:
        attempts.append(base + ['-DUSE_BLAS', '-lopenblas', '-lm'])
    attempts.append(base + ['-lm'])
    last = None
    for cmd in attempts:
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return out
        except subprocess.CalledProcessError as e:
            last = e.stderr.decode('utf-8', 'replace') if e.stderr else str(e)
    raise RuntimeError(f'failed to build libnotorch:\n{last}')


_lib = ctypes.CDLL(_find_or_build_lib())

_V = ctypes.c_void_p
_I = ctypes.c_int
_F = ctypes.c_float


def _sig(name, restype, *argtypes):
    fn = getattr(_lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)


# tensor
_sig('nt_tensor_new', _V, _I)
_sig('nt_tensor_new2d', _V, _I, _I)
_sig('nt_tensor_free', None, _V)
_sig('nt_tensor_fill', None, _V, _F)
_sig('nt_tensor_xavier', None, _V, _I, _I)
_sig('nt_tensor_rand', None, _V, _F)
_sig('nt_seed', None, ctypes.c_uint64)
# tape
_sig('nt_tape_start', None)
_sig('nt_tape_clear', None)
_sig('nt_tape_get', _V)
_sig('nt_tape_param', _I, _V)
_sig('nt_tape_backward', None, _I)
_sig('nt_tape_clip_grads', _F, _F)
_sig('nt_tape_chuck_step', None, _F, _F)
_sig('nt_tape_record', _I, _V, _I, _I, _I, _F)
_sig('nt_train_mode', None, _I)
# ops
_sig('nt_seq_embedding', _I, _I, _I, _I, _I, _I)
_sig('nt_seq_rmsnorm', _I, _I, _I, _I, _I)
_sig('nt_seq_linear', _I, _I, _I, _I)
_sig('nt_mh_causal_attention', _I, _I, _I, _I, _I, _I)
_sig('nt_rrpram_attention', _I, _I, _I, _I, _I, _I, _I, _I)
_sig('nt_seq_gate', _I, _I, _I, _I, _I, _I)
_sig('nt_sigmoid', _I, _I)
_sig('nt_mul', _I, _I, _I)
_sig('nt_scale', _I, _I, _F)
_sig('nt_concat', _I, _I, _I, _I)
_sig('nt_relu', _I, _I)
_sig('nt_add', _I, _I, _I)
_sig('nt_seq_cross_entropy', _I, _I, _I, _I, _I)


class _NtTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(_F)), ("ndim", _I),
        ("shape", _I * 8), ("stride", _I * 8), ("len", _I), ("refcount", _I),
    ]


class _NtTapeEntry(ctypes.Structure):
    _fields_ = [
        ("output", _V), ("grad", _V),
        ("op", _I), ("parent1", _I), ("parent2", _I), ("parent3", _I),
        ("aux", _F), ("aux2", _F), ("aux3", _F), ("aux4", _F),
        ("is_param", _I), ("no_decay", _I), ("frozen", _I),
    ]


def _tstruct(ptr):
    return ctypes.cast(ptr, ctypes.POINTER(_NtTensor)).contents


def _entry_output(idx):
    tape = _lib.nt_tape_get()
    e = ctypes.cast(tape, ctypes.POINTER(_NtTapeEntry))[idx]
    return _tstruct(e.output)


class Tensor:
    def __init__(self, ptr, owns=True):
        self._ptr = ptr
        self._owns = owns

    @staticmethod
    def zeros(size):
        if isinstance(size, int):
            ptr = _lib.nt_tensor_new(size)
        elif len(size) == 1:
            ptr = _lib.nt_tensor_new(size[0])
        else:
            ptr = _lib.nt_tensor_new2d(size[0], size[1])
        return Tensor(ptr)

    @staticmethod
    def ones(size):
        t = Tensor.zeros(size)
        _lib.nt_tensor_fill(t._ptr, 1.0)
        return t

    @property
    def numel(self):
        return _tstruct(self._ptr).len

    def fill_(self, v):
        _lib.nt_tensor_fill(self._ptr, _F(v)); return self

    def xavier_(self, fi, fo):
        _lib.nt_tensor_xavier(self._ptr, fi, fo); return self

    def rand_(self, s):
        _lib.nt_tensor_rand(self._ptr, _F(s)); return self

    def get_data(self):
        s = _tstruct(self._ptr)
        return [s.data[i] for i in range(s.len)]

    def __del__(self):
        if getattr(self, '_owns', False) and getattr(self, '_ptr', None):
            _lib.nt_tensor_free(self._ptr)


def softmax(logits, temperature=1.0):
    if temperature != 1.0:
        logits = [x / temperature for x in logits]
    mx = max(logits)
    e = [math.exp(x - mx) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def multinomial(probs):
    r = random.random(); c = 0.0
    for i, p in enumerate(probs):
        c += p
        if c >= r:
            return i
    return len(probs) - 1


def seed(s):
    _lib.nt_seed(ctypes.c_uint64(s))


# ═══════════════════════════════════════════════════════════════════════════════
# Q ENGINE — builds PostGPT-Q's ε transformer (triple gated attention) on the tape
# ═══════════════════════════════════════════════════════════════════════════════

class QEngine:
    """
    Forward/backward/update for PostGPT-Q's transformer substrate on the notorch tape.

    cfg: V, D, NL, CTX, NC, NR, NJ, HD, nm (mechanism count).
    params: flat ordered list, per QPTQ layout:
        [tok, pos,
         (per layer: [wq,wk,vc if NC] , [wr,vr if NR] , [wj,vj if NJ] ,
          [gws_aug [nm,D+1] if nm>1 — last column is the trained gate bias gb] ,
          wo, up, dn)]
    lm_head is TIED to tok. Norm is parameter-free (fixed ones gamma).
    """

    def __init__(self, params, cfg, optimizer):
        self.params = params
        self.cfg = cfg
        self.optimizer = optimizer
        D = cfg['D']
        self.ones_D = Tensor.ones(D)                       # param-free rmsnorm gamma
        self.ones_J = Tensor.ones(cfg['NJ'] * cfg['HD']) if cfg['NJ'] > 0 else None
        self.gb_inv_sqrt = 1.0 / math.sqrt(cfg['NJ'] * cfg['HD']) if cfg['NJ'] > 0 else 0.0

    def _build(self, tok_ids, tgt_ids):
        c = self.cfg
        T = len(tok_ids); D = c['D']; HD = c['HD']
        NC = c['NC']; NR = c['NR']; NJ = c['NJ']; nm = c['nm']; V = c['V']; L = c['NL']

        _lib.nt_tape_start()
        tp = [_lib.nt_tape_param(p._ptr) for p in self.params]
        # constant tensors recorded on the tape (NT_OP_NONE = 0)
        gD = _lib.nt_tape_record(self.ones_D._ptr, 0, -1, -1, _F(0))
        gJ = _lib.nt_tape_record(self.ones_J._ptr, 0, -1, -1, _F(0)) if NJ > 0 else -1

        tok_t = Tensor.zeros(T);
        for i, v in enumerate(tok_ids):
            _tstruct(tok_t._ptr).data[i] = float(v)
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, _F(0))
        # constant 1.0 channel for the gate-bias augment (gb trains as the last gws column)
        ones_t = Tensor.ones(T)
        ones_idx = _lib.nt_tape_record(ones_t._ptr, 0, -1, -1, _F(0))
        if tgt_ids is not None:
            tgt_t = Tensor.zeros(T)
            for i, v in enumerate(tgt_ids):
                _tstruct(tgt_t._ptr).data[i] = float(v)
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, _F(0))

        pi = 0
        tok_w = tp[pi]; pi += 1
        pos_w = tp[pi]; pi += 1
        h = _lib.nt_seq_embedding(tok_w, pos_w, tok_idx, T, D)

        for _ in range(L):
            xn = _lib.nt_seq_rmsnorm(h, gD, T, D)
            blocks = []   # (tape_idx, block_size, gate_col)
            gi = 0
            if NC > 0:
                q = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                k = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                vc = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                co = _lib.nt_mh_causal_attention(q, k, vc, T, HD)
                blocks.append((co, NC * HD, gi)); gi += 1
            if NR > 0:
                wr = tp[pi]; pi += 1
                vr = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                ro = _lib.nt_rrpram_attention(wr, xn, vr, T, D, NR, HD)
                blocks.append((ro, NR * HD, gi)); gi += 1
            if NJ > 0:
                wjp = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                vjp = _lib.nt_seq_linear(tp[pi], xn, T); pi += 1
                # jo = vjp ⊙ (wjp / ||wjp||) = vjp ⊙ (rmsnorm(wjp) / sqrt(NJ*HD))
                wjn = _lib.nt_seq_rmsnorm(wjp, gJ, T, NJ * HD)
                wjn = _lib.nt_scale(wjn, _F(self.gb_inv_sqrt))
                jo = _lib.nt_mul(vjp, wjn)
                blocks.append((jo, NJ * HD, gi)); gi += 1

            # gating: sigmoid(gws@xn + gb). gb is trained as the last column of an
            # augmented gws [nm, D+1] applied to xn augmented with a constant-1 channel.
            if nm > 1:
                gws = tp[pi]; pi += 1                       # gws_aug [nm, D+1] (last col = gb)
                xn_aug = _lib.nt_concat(xn, ones_idx, T)    # [T, D+1]
                gl = _lib.nt_seq_linear(gws, xn_aug, T)     # [T, nm] = gws@xn + gb
                gate = _lib.nt_sigmoid(gl)                  # [T, nm]
                gated = [_lib.nt_seq_gate(b, gate, T, nm, col) for (b, _bs, col) in blocks]
            else:
                gated = [b for (b, _bs, _col) in blocks]

            # concat all mechanism blocks -> [T, D]
            comb = gated[0]
            for nxt in gated[1:]:
                comb = _lib.nt_concat(comb, nxt, T)

            wo = tp[pi]; pi += 1
            proj = _lib.nt_seq_linear(wo, comb, T)
            h = _lib.nt_add(h, proj)

            # MLP: dn(relu(up(rmsnorm(h))))
            xn2 = _lib.nt_seq_rmsnorm(h, gD, T, D)
            up = _lib.nt_seq_linear(tp[pi], xn2, T); pi += 1
            up = _lib.nt_relu(up)
            dn = _lib.nt_seq_linear(tp[pi], up, T); pi += 1
            h = _lib.nt_add(h, dn)

        hf = _lib.nt_seq_rmsnorm(h, gD, T, D)
        logits = _lib.nt_seq_linear(tok_w, hf, T)         # tied lm_head
        if tgt_ids is None:
            return logits
        return _lib.nt_seq_cross_entropy(logits, tgt_idx, T, V)

    def step(self, tok_ids, tgt_ids):
        loss_idx = self._build(tok_ids, tgt_ids)
        loss = _entry_output(loss_idx).data[0]
        _lib.nt_tape_backward(loss_idx)
        self.optimizer.step(loss)
        _lib.nt_tape_clear()
        return loss

    def logits_last(self, tok_ids):
        V = self.cfg['V']; T = len(tok_ids)
        li = self._build(tok_ids, None)
        out = _entry_output(li)
        base = (T - 1) * V
        vals = [out.data[base + j] for j in range(V)]
        _lib.nt_tape_clear()
        return vals


__all__ = ['Tensor', 'QEngine', 'softmax', 'multinomial', 'seed', '_lib', '_tstruct']
