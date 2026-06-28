"""
train_q_notorch.py — train PostGPT-Q's ε substrate from scratch on notorch (no PyTorch).

Q's weights were originally trained on an A100 with torch. This trains the same
transformer substrate (triple gated attention: Content + RRPRAM + Janus echo) on the
notorch autograd tape via the ariannamethod/ ctypes shim, with the real Chuck
optimizer, and exports the QPTQ .bin that postgpt_q.c / .py / .html load directly.

The runtime, the tokenizer (q.merges), and the .bin format are unchanged — only the
training path is now torch-free. q-coherence-from-scratch, trained from scratch.

Usage:
    python3 train_q_notorch.py [--steps N] [--variant rrpram3_janus3|rrpram_6r] [--save out.bin]

Note (v1): the per-mechanism gate bias (gbs) is exported as zeros (see notorch_nn.py).
The magnitude "transformer gate" is inference-only (applied by postgpt_q.c), so training
optimizes raw logits — q stays silent at inference until weights earn magnitude, by design.
"""

import os
import sys
import math
import time
import struct
import random
import argparse

# Reuse q's own BPE (import-safe: postgpt_q.py guards main) so vocab matches inference.
import postgpt_q as Q
from ariannamethod import QEngine, ChuckOptimizer, Tensor, softmax, multinomial, seed

QPTQ_MAGIC = 0x51505451
QPTQ_VERSION = 1

VARIANTS = {
    "rrpram3_janus3": dict(NC=0, NR=3, NJ=3),
    "rrpram_6r":      dict(NC=0, NR=6, NJ=0),
}


def make_cfg(variant, V, D=192, NL=3, CTX=128, NH=6):
    v = VARIANTS[variant]
    NC, NR, NJ = v["NC"], v["NR"], v["NJ"]
    assert NC + NR + NJ == NH, "head layout must sum to NH"
    HD = D // NH
    nm = (1 if NC > 0 else 0) + (1 if NR > 0 else 0) + (1 if NJ > 0 else 0)
    return dict(V=V, D=D, NH=NH, NL=NL, CTX=CTX, NC=NC, NR=NR, NJ=NJ, HD=HD, nm=nm)


def build_params(cfg):
    V, D, CTX = cfg["V"], cfg["D"], cfg["CTX"]
    NC, NR, NJ, HD, nm, L = cfg["NC"], cfg["NR"], cfg["NJ"], cfg["HD"], cfg["nm"], cfg["NL"]

    def lin(out_f, in_f):
        t = Tensor.zeros((out_f, in_f)); t.xavier_(in_f, out_f); return t

    P = []
    tok = Tensor.zeros((V, D)); tok.rand_(0.02); P.append(tok)
    pos = Tensor.zeros((CTX, D)); pos.rand_(0.02); P.append(pos)
    for _ in range(L):
        if NC > 0:
            P += [lin(NC * HD, D), lin(NC * HD, D), lin(NC * HD, D)]  # wq, wk, vc
        if NR > 0:
            wr = Tensor.zeros(NR * D * CTX); wr.rand_(0.02); P.append(wr)  # wrs [NR,D,CTX]
            P.append(lin(NR * HD, D))                                       # vr
        if NJ > 0:
            P.append(lin(NJ * HD, D))   # wj
            P.append(lin(NJ * HD, D))   # vj
        if nm > 1:
            # gws_aug [nm, D+1]: columns 0..D-1 = gate weight, column D = gate bias (gb)
            gws = Tensor.zeros((nm, D + 1)); gws.rand_(0.02); P.append(gws)
        P.append(lin(D, D))         # wo
        P.append(lin(4 * D, D))     # up
        P.append(lin(D, 4 * D))     # dn
    return P


def export_qptq(params, cfg, path):
    """Write the QPTQ .bin — byte-identical layout to tools/export_q_weights.py."""
    V, D, NH, NL, CTX = cfg["V"], cfg["D"], cfg["NH"], cfg["NL"], cfg["CTX"]
    NC, NR, NJ, HD, nm = cfg["NC"], cfg["NR"], cfg["NJ"], cfg["HD"], cfg["nm"]
    state = {"i": 0}

    def nxt():
        t = params[state["i"]]; state["i"] += 1; return t.get_data()

    def wf(f, vals):
        f.write(struct.pack("<%df" % len(vals), *vals))

    with open(path, "wb") as f:
        f.write(struct.pack("<I", QPTQ_MAGIC))
        f.write(struct.pack("<10I", QPTQ_VERSION, V, D, NH, NL, CTX, NC, NR, NJ, HD))
        wf(f, nxt())  # tok
        wf(f, nxt())  # pos
        for _ in range(NL):
            if NC > 0:
                wf(f, nxt()); wf(f, nxt()); wf(f, nxt())   # wq, wk, vc
            if NR > 0:
                wf(f, nxt()); wf(f, nxt())                 # wrs, vr
            if NJ > 0:
                wf(f, nxt()); wf(f, nxt())                 # wj, vj
            if nm > 1:
                ga = nxt()                                 # gws_aug flat [nm*(D+1)]
                gw_rows = []; gbs = []
                for g in range(nm):
                    row = ga[g * (D + 1):(g + 1) * (D + 1)]
                    gw_rows.extend(row[:D]); gbs.append(row[D])
                wf(f, gw_rows)                             # gws [nm, D]
                wf(f, gbs)                                 # gbs [nm] (trained, not zeros)
            wf(f, nxt())  # wo
            wf(f, nxt())  # up
            wf(f, nxt())  # dn
    assert state["i"] == len(params), f"param/export mismatch: used {state['i']} of {len(params)}"


def get_sequence(ids, seq_len):
    i = random.randint(0, len(ids) - seq_len - 1)
    return ids[i:i + seq_len], ids[i + 1:i + seq_len + 1]


def train(args):
    base = os.path.dirname(os.path.abspath(__file__))
    merges_path = os.path.join(base, "q.merges")
    corpus_path = os.path.join(base, "q.txt")
    for p in (merges_path, corpus_path):
        if not os.path.exists(p):
            print("ERROR: missing " + p); return

    seed(42); random.seed(42)

    print("[1] BPE + tokenize q.txt ...")
    bpe = Q.BPE()
    if not Q.bpe_load(bpe, merges_path):
        print("ERROR: bpe_load failed"); return
    raw = open(corpus_path, "rb").read()
    ids = Q.bpe_encode(bpe, raw, len(raw), len(raw))
    print(f"  tokens={len(ids)} vocab={bpe.vocab_size}")

    cfg = make_cfg(args.variant, bpe.vocab_size, CTX=args.ctx)
    print(f"[2] build params ({args.variant}): "
          f"V={cfg['V']} D={cfg['D']} NL={cfg['NL']} NR={cfg['NR']} NJ={cfg['NJ']} HD={cfg['HD']} nm={cfg['nm']}")
    params = build_params(cfg)
    nparams = sum(p.numel for p in params)
    print(f"  {nparams:,} params, {len(params)} tensors")

    optimizer = ChuckOptimizer(lr=args.lr, max_grad_norm=1.0)
    engine = QEngine(params, cfg, optimizer)

    print(f"[3] train {args.steps} steps (seq_len={args.seq_len}) ...")
    losses = []; t0 = time.time()
    for step in range(args.steps):
        x, y = get_sequence(ids, args.seq_len)
        loss = engine.step(x, y)
        losses.append(loss)
        if (step + 1) % 10 == 0 or step == 0:
            avg = sum(losses[-10:]) / len(losses[-10:])
            print(f"  step {step+1:4d}/{args.steps}  loss={loss:.4f}  avg10={avg:.4f}  [{time.time()-t0:.1f}s]")

    first = sum(losses[:10]) / min(10, len(losses))
    last = sum(losses[-10:]) / min(10, len(losses))
    print(f"\n  first10={first:.4f}  last10={last:.4f}  delta={last-first:.4f}  (uniform floor ln(V)={math.log(cfg['V']):.3f})")

    if args.save:
        export_qptq(params, cfg, args.save)
        print(f"[4] exported QPTQ -> {args.save} ({os.path.getsize(args.save)/1024:.1f} KB)")

    print("\n  done. notorch trained q's substrate. no torch.")
    return losses


def main():
    ap = argparse.ArgumentParser(description="Train PostGPT-Q substrate on notorch (no PyTorch)")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--variant", type=str, default="rrpram3_janus3", choices=list(VARIANTS))
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--save", type=str, default="")
    args = ap.parse_args()
    print("=" * 60)
    print("  PostGPT-Q substrate training on notorch + Chuck")
    print("=" * 60)
    train(args)


if __name__ == "__main__":
    main()
