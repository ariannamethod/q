"""
postgpt_q.py — PostGPT-Q: Resonant Reasoning Engine

Built on PostGPT. Val autograd, dual attention (Content + RRPRAM),
metaweight seeding, Dario equation.

The transformer provides rhythm. The metaweights provide voice.
The Dario equation provides physics. The chambers provide body.

The transformer does NOT speak. It modulates logits.
PostGPT mechanism speaks — it jumps to what resonates.
12 bidirectional sentences, not prompt continuation.
Interference via sentence-boundary injection from docs/.

Zero dependencies. math, random, os, struct, time, re.

(c) 2026 arianna method
resonance is unbreakable.
"""

import math
import os
import random
import re
import struct
import sys
import time

random.seed(42)


# ─────────────────────────────────────────────────────────────────
# I. BPE TOKENIZER — from PostGPT, identical.
# ─────────────────────────────────────────────────────────────────

class BPETokenizer:
    def __init__(self, max_merges=1024):
        self.max_merges = max_merges
        self.merges = []
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}

    def _count_pairs(self, ids):
        counts = {}
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, new_id):
        result = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(ids[i])
                i += 1
        return result

    def learn(self, data_bytes, num_merges=None):
        if num_merges is None:
            num_merges = self.max_merges
        ids = list(data_bytes)
        for m in range(num_merges):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            new_id = 256 + m
            ids = self._merge_pair(ids, best, new_id)
            self.merges.append((best[0], best[1], new_id))
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            self.vocab_size = 256 + m + 1
        return ids

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge_pair(ids, (a, b), new_id)
        return ids

    def decode(self, ids):
        raw = b''
        for tid in ids:
            if tid in self.vocab:
                raw += self.vocab[tid]
        return raw.decode('utf-8', errors='replace')

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(struct.pack('<I', len(self.merges)))
            for a, b, new_id in self.merges:
                f.write(struct.pack('<III', a, b, new_id))

    def load(self, path):
        with open(path, 'rb') as f:
            n = struct.unpack('<I', f.read(4))[0]
            self.merges = []
            for _ in range(n):
                a, b, new_id = struct.unpack('<III', f.read(12))
                self.merges.append((a, b, new_id))
                self.vocab[new_id] = self.vocab.get(a, bytes([a % 256])) + \
                                     self.vocab.get(b, bytes([b % 256]))
            self.vocab_size = 256 + len(self.merges)


# ─────────────────────────────────────────────────────────────────
# II. METAWEIGHTS — from PostGPT, identical.
# ─────────────────────────────────────────────────────────────────

class MetaWeights:
    def __init__(self, vocab_size, context_len):
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.unigram = [0.0] * vocab_size
        self.bigram = {}
        self.trigram = {}
        self.pos_affinity = {}
        self.hebbian = {}
        self.total = 0

    def build(self, token_ids, window=8):
        n = len(token_ids)
        self.total = n
        for tid in token_ids:
            if tid < self.vocab_size:
                self.unigram[tid] += 1.0
        total = sum(self.unigram)
        if total > 0:
            self.unigram = [c / total for c in self.unigram]
        for i in range(n - 1):
            a, b = token_ids[i], token_ids[i + 1]
            if a not in self.bigram:
                self.bigram[a] = {}
            self.bigram[a][b] = self.bigram[a].get(b, 0) + 1
        for a in self.bigram:
            t = sum(self.bigram[a].values())
            if t > 0:
                for b in self.bigram[a]:
                    self.bigram[a][b] /= t
        for i in range(n - 2):
            key = (token_ids[i], token_ids[i + 1])
            c = token_ids[i + 2]
            if key not in self.trigram:
                self.trigram[key] = {}
            self.trigram[key][c] = self.trigram[key].get(c, 0) + 1
        for key in self.trigram:
            t = sum(self.trigram[key].values())
            if t > 0:
                for c in self.trigram[key]:
                    self.trigram[key][c] /= t
        for i in range(n):
            pos = i % self.context_len
            tid = token_ids[i]
            if tid not in self.pos_affinity:
                self.pos_affinity[tid] = [0.0] * self.context_len
            self.pos_affinity[tid][pos] += 1.0
        for tid in self.pos_affinity:
            t = sum(self.pos_affinity[tid])
            if t > 0:
                self.pos_affinity[tid] = [c / t for c in self.pos_affinity[tid]]
        hebb_n = min(n, 20000)
        for i in range(hebb_n):
            for j in range(max(0, i - window), min(hebb_n, i + window + 1)):
                if i == j:
                    continue
                a, b = token_ids[i], token_ids[j]
                key = (min(a, b), max(a, b))
                decay = 1.0 / (1.0 + abs(i - j))
                self.hebbian[key] = self.hebbian.get(key, 0.0) + decay
        if self.hebbian:
            mx = max(self.hebbian.values())
            if mx > 0:
                for key in self.hebbian:
                    self.hebbian[key] /= mx

    def query_bigram(self, prev, V):
        d = [1e-10] * V
        if prev in self.bigram:
            for tok, p in self.bigram[prev].items():
                if tok < V:
                    d[tok] = p
        return d

    def query_trigram(self, p2, p1, V):
        d = [1e-10] * V
        key = (p2, p1)
        if key in self.trigram:
            for tok, p in self.trigram[key].items():
                if tok < V:
                    d[tok] = p
        return d

    def query_hebbian(self, ctx, V):
        s = [0.0] * V
        for (a, b), st in self.hebbian.items():
            for c in ctx:
                if a == c and b < V:
                    s[b] += st
                elif b == c and a < V:
                    s[a] += st
        mx = max(s) if s else 1.0
        if mx > 0:
            s = [x / mx for x in s]
        return s

    def query_prophecy(self, ctx, V, top_k=16):
        appeared = set(ctx)
        s = [0.0] * V
        for c in ctx[-4:]:
            if c in self.bigram:
                for tok, p in sorted(self.bigram[c].items(),
                                     key=lambda x: -x[1])[:top_k]:
                    if tok not in appeared and tok < V:
                        s[tok] += p
        mx = max(s) if s else 1.0
        if mx > 0:
            s = [x / mx for x in s]
        return s


# ─────────────────────────────────────────────────────────────────
# III. VAL AUTOGRAD — from PostGPT, identical.
# ─────────────────────────────────────────────────────────────────

class Val:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, o):
        o = o if isinstance(o, Val) else Val(o)
        return Val(self.data + o.data, (self, o), (1.0, 1.0))
    def __mul__(self, o):
        o = o if isinstance(o, Val) else Val(o)
        return Val(self.data * o.data, (self, o), (o.data, self.data))
    def __pow__(self, o):
        return Val(self.data ** o, (self,), (o * self.data ** (o - 1),))
    def log(self):
        d = max(self.data, 1e-12)
        return Val(math.log(d), (self,), (1.0 / d,))
    def exp(self):
        e = math.exp(min(self.data, 80))
        return Val(e, (self,), (e,))
    def relu(self):
        return Val(max(0, self.data), (self,), (float(self.data > 0),))
    def tanh(self):
        t = math.tanh(self.data)
        return Val(t, (self,), (1.0 - t * t,))

    def __neg__(self): return self * -1
    def __radd__(self, o): return self + o
    def __sub__(self, o): return self + (-o)
    def __rsub__(self, o): return (-self) + o
    def __rmul__(self, o): return self * o
    def __truediv__(self, o):
        return self * (o if isinstance(o, Val) else Val(o)) ** -1
    def __rtruediv__(self, o): return Val(o) * self ** -1

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# ─────────────────────────────────────────────────────────────────
# IV. TRANSFORMER — from PostGPT. Val-based. Content + RRPRAM.
# ─────────────────────────────────────────────────────────────────

def _randn(std=0.02): return random.gauss(0, std)
def _matrix(r, c, std=0.02): return [[Val(_randn(std)) for _ in range(c)] for _ in range(r)]
def linear(x, w): return [sum(wi*xi for wi,xi in zip(row,x)) for row in w]
def softmax_val(logits):
    mx = max(v.data for v in logits)
    exps = [(v - mx).exp() for v in logits]
    t = sum(exps)
    return [e / t for e in exps]
def softmax_float(logits):
    mx = max(logits)
    exps = [math.exp(min(v - mx, 80)) for v in logits]
    t = sum(exps)
    return [e / t for e in exps]
def rmsnorm(x):
    ms = sum(xi*xi for xi in x) / len(x)
    s = (ms + Val(1e-5)) ** -0.5
    return [xi * s for xi in x]


def apply_rope(q, k, pos, head_dim):
    """Rotary Position Embedding — relative position awareness.
    Rotates Q and K vectors by position-dependent angles.
    RoPE + RRPRAM = relative + absolute position = full picture."""
    half = head_dim // 2
    q_rot, k_rot = list(q), list(k)
    for i in range(half):
        freq = 1.0 / (10000.0 ** (2.0 * i / head_dim))
        angle = pos * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        # rotate pairs (2i, 2i+1)
        q0 = q[2*i].data if isinstance(q[2*i], Val) else q[2*i]
        q1 = q[2*i+1].data if isinstance(q[2*i+1], Val) else q[2*i+1]
        k0 = k[2*i].data if isinstance(k[2*i], Val) else k[2*i]
        k1 = k[2*i+1].data if isinstance(k[2*i+1], Val) else k[2*i+1]
        q_rot[2*i] = Val(q0 * cos_a - q1 * sin_a)
        q_rot[2*i+1] = Val(q0 * sin_a + q1 * cos_a)
        k_rot[2*i] = Val(k0 * cos_a - k1 * sin_a)
        k_rot[2*i+1] = Val(k0 * sin_a + k1 * cos_a)
    return q_rot, k_rot


class QTransformer:
    """Dual attention transformer. Val autograd. RoPE + RRPRAM."""

    def __init__(self, vocab_size, ctx=64, embd=48, n_head=4,
                 n_layer=2, n_content=2, n_rrpram=2, n_janus=0):
        self.vocab_size = vocab_size
        self.context_len = ctx
        self.n_embd = embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_content = n_content
        self.n_rrpram = n_rrpram
        self.n_janus = n_janus  # Janus echo (self-resonance) heads
        self.head_dim = embd // n_head
        assert n_content + n_rrpram + n_janus == n_head
        hd = self.head_dim
        n_mech = (1 if n_content > 0 else 0) + (1 if n_rrpram > 0 else 0) + (1 if n_janus > 0 else 0)

        self.wte = _matrix(vocab_size, embd)
        self.wpe = _matrix(ctx, embd)
        self.layers = []
        for _ in range(n_layer):
            layer = {
                'wq': _matrix(n_content * hd, embd),
                'wk': _matrix(n_content * hd, embd),
                'wv_c': _matrix(n_content * hd, embd),
                'wr': _matrix(n_rrpram * embd, ctx),
                'wv_r': _matrix(n_rrpram * hd, embd),
                # Janus echo: W_j for self-resonance (W^T·W·x)
                'wj': _matrix(n_janus * hd, embd) if n_janus > 0 else [],
                'wv_j': _matrix(n_janus * hd, embd) if n_janus > 0 else [],
                # Learned gate: sigmoid per mechanism type (pitomadom-style)
                'gate_w': _matrix(n_mech, embd, std=0.01),
                'gate_b': [Val(0.0) for _ in range(n_mech)],
                'wo': _matrix(embd, embd, std=0.02/math.sqrt(2*n_layer)),
                'mlp_up': _matrix(4*embd, embd),
                'mlp_down': _matrix(embd, 4*embd, std=0.02/math.sqrt(2*n_layer)),
            }
            self.layers.append(layer)
        self.lm_head = _matrix(vocab_size, embd)
        self.destiny = [0.0] * embd
        self.trauma = 0.0
        self.temperature = 0.85

        self.params = []
        for row in self.wte: self.params.extend(row)
        for row in self.wpe: self.params.extend(row)
        for L in self.layers:
            for k in L:
                v = L[k]
                if isinstance(v, list) and len(v) > 0:
                    if isinstance(v[0], list):
                        for row in v: self.params.extend(row)
                    elif isinstance(v[0], Val):
                        self.params.extend(v)
        for row in self.lm_head: self.params.extend(row)
        print(f"  QTransformer: {len(self.params)} params, v={vocab_size}, "
              f"e={embd}, h={n_head}({n_content}c+{n_rrpram}r), L={n_layer}")

    def init_from_metaweights(self, meta):
        V, E, T = self.vocab_size, self.n_embd, self.context_len
        sc = 0.15
        for a in range(min(V, len(self.wte))):
            sig = [0.0]*E; nn = 0
            for b in range(min(V, len(self.wte))):
                key = (min(a,b), max(a,b))
                if key in meta.hebbian and meta.hebbian[key] > 0.01:
                    for d in range(E): sig[d] += meta.hebbian[key] * self.wte[b][d].data
                    nn += 1
            if nn > 0:
                for d in range(E): self.wte[a][d].data += sc * sig[d] / nn
        for pos in range(min(T, len(self.wpe))):
            sig = [0.0]*E; nt = 0
            for tok in meta.pos_affinity:
                if tok < V and pos < len(meta.pos_affinity[tok]):
                    aff = meta.pos_affinity[tok][pos]
                    if aff > 0.001:
                        for d in range(E): sig[d] += aff * self.wte[tok][d].data
                        nt += 1
            if nt > 0:
                for d in range(E): self.wpe[pos][d].data += sc * sig[d] / nt
        for L in self.layers:
            wr = L['wr']
            for h in range(self.n_rrpram):
                for tok in meta.pos_affinity:
                    if tok >= V: continue
                    affs = meta.pos_affinity[tok]
                    for pos in range(min(T, len(affs))):
                        if affs[pos] > 0.001:
                            r = h*E + (tok % E)
                            if r < len(wr) and pos < len(wr[r]):
                                wr[r][pos].data += sc * 0.5 * affs[pos]
        for tok in range(min(V, len(self.lm_head))):
            freq = meta.unigram[tok] if tok < len(meta.unigram) else 0
            if freq > 0:
                for d in range(E):
                    self.lm_head[tok][d].data += sc * freq * self.wte[tok][d].data

    def forward_token(self, token_id, pos_id, kv_cache):
        hd = self.head_dim
        nc, nr, nj = self.n_content, self.n_rrpram, self.n_janus
        x = [t+p for t,p in zip(self.wte[token_id], self.wpe[pos_id])]

        for li in range(self.n_layer):
            L = self.layers[li]
            kc, vcc, vrc, jc = kv_cache[li]
            xr = x; xn = rmsnorm(x)

            q = linear(xn, L['wq']); k = linear(xn, L['wk'])
            vc = linear(xn, L['wv_c']); vr = linear(xn, L['wv_r'])

            # RoPE on content Q,K
            for h in range(nc):
                hs = h*hd
                qh_r, kh_r = apply_rope(q[hs:hs+hd], k[hs:hs+hd], pos_id, hd)
                for d in range(hd):
                    q[hs+d] = qh_r[d]; k[hs+d] = kh_r[d]
            kc.append(k); vcc.append(vc); vrc.append(vr)

            # === MECHANISM 1: Content attention (QK^T + RoPE) ===
            content_out = []
            for h in range(nc):
                hs = h*hd
                qh = q[hs:hs+hd]
                ka = [ki[hs:hs+hd] for ki in kc]
                va = [vi[hs:hs+hd] for vi in vcc]
                al = [sum(qh[j]*ka[t][j] for j in range(hd)) * (1.0/math.sqrt(hd))
                      for t in range(len(ka))]
                aw = softmax_val(al)
                content_out.extend([sum(aw[t]*va[t][j] for t in range(len(va)))
                                    for j in range(hd)])

            # === MECHANISM 2: RRPRAM (x @ Wr — positional) ===
            rrpram_out = []
            for h in range(nr):
                hs = h*hd
                wo = h*self.n_embd
                wrh = L['wr'][wo:wo+self.n_embd]
                sl = len(kc)
                al = []
                for t in range(sl):
                    sc = Val(0.0)
                    for d in range(min(self.n_embd, len(wrh))):
                        if t < len(wrh[d]): sc = sc + xn[d] * wrh[d][t]
                    al.append(sc)
                aw = softmax_val(al) if al else []
                va = [vi[hs:hs+hd] for vi in vrc]
                ho = []
                for j in range(hd):
                    vs = Val(0.0)
                    for t in range(len(aw)):
                        if t < len(va): vs = vs + aw[t] * va[t][j]
                    ho.append(vs)
                rrpram_out.extend(ho)

            # === MECHANISM 3: Janus echo (W^T·W self-resonance) ===
            # Real MetaJanus: echo(x,W) = x · (W^T · W · x) / ||W·x||
            # Each position computes its echo. Attention = echo_i · echo_j.
            # Positions that the model recognizes similarly → attend to each other.
            janus_out = []
            if nj > 0 and L['wj']:
                wj_curr = linear(xn, L['wj'])  # [nj*hd]
                jc.append(wj_curr)  # cache janus projections

                vj = linear(xn, L['wv_j'])  # value projection

                for h in range(nj):
                    hs = h * hd
                    # current echo: ||W·x_curr||
                    curr_proj = wj_curr[hs:hs+hd]
                    curr_norm = sum(w*w for w in curr_proj)
                    curr_echo = (curr_norm + Val(1e-8)) ** 0.5

                    # echo(x) = x · (W^T · W · x) / ||W·x||
                    # = (W·x)^T · (W·x) / ||W·x|| = ||W·x||
                    # attention[i,j] = echo_i · echo_j (mutual resonance)
                    attn_scores = []
                    for t in range(len(jc)):
                        cached = jc[t][hs:hs+hd]
                        cached_norm = sum(c*c for c in cached)
                        cached_echo = (cached_norm + Val(1e-8)) ** 0.5
                        # mutual resonance
                        score = curr_echo * cached_echo
                        attn_scores.append(score)

                    if attn_scores:
                        aw = softmax_val(attn_scores)
                        # aggregate values weighted by mutual resonance
                        ho = []
                        for j in range(hd):
                            vs = Val(0.0)
                            for t in range(len(aw)):
                                if t < len(jc):
                                    # use cached projections as values
                                    vs = vs + aw[t] * jc[t][hs + j]
                            ho.append(vs)
                        janus_out.extend(ho)
                    else:
                        janus_out.extend(curr_proj)

            # === GATING: learned sigmoid per mechanism (pitomadom-style) ===
            mechanisms = []
            if content_out: mechanisms.append(content_out)
            if rrpram_out: mechanisms.append(rrpram_out)
            if janus_out: mechanisms.append(janus_out)

            if len(mechanisms) > 1 and L['gate_w']:
                # compute gate values: sigmoid(gate_w @ xn + gate_b)
                gate_logits = linear(xn, L['gate_w'])
                gates = []
                for gi in range(len(mechanisms)):
                    if gi < len(gate_logits) and gi < len(L['gate_b']):
                        g = (gate_logits[gi] + L['gate_b'][gi])
                        # sigmoid via Val ops
                        g_val = ((-g).exp() + Val(1.0)) ** -1
                        gates.append(g_val)
                    else:
                        gates.append(Val(1.0))

                # apply gates and concatenate
                xa = []
                for gi, mech in enumerate(mechanisms):
                    g = gates[gi] if gi < len(gates) else Val(1.0)
                    xa.extend([g * v for v in mech])
            else:
                # single mechanism or no gating
                xa = []
                for mech in mechanisms:
                    xa.extend(mech)

            xp = linear(xa, L['wo'])
            x = [a+b for a,b in zip(xp, xr)]

            # MLP
            xr = x; xn = rmsnorm(x)
            hm = [hi.relu() for hi in linear(xn, L['mlp_up'])]
            xm = linear(hm, L['mlp_down'])
            x = [a+b for a,b in zip(xm, xr)]

        x = rmsnorm(x)
        return linear(x, self.lm_head)

    def generate(self, prompt_ids, max_tokens=64, meta=None, temp=None):
        if temp is None: temp = self.temperature
        kv = [([], [], [], []) for _ in range(self.n_layer)]
        gen = list(prompt_ids); ctx = list(prompt_ids)
        for pos, tid in enumerate(prompt_ids):
            if pos >= self.context_len - 1: break
            self.forward_token(tid, pos, kv)
        for step in range(max_tokens):
            pos = len(ctx) - 1
            if pos >= self.context_len - 1: break
            logits = self.forward_token(ctx[-1], pos, kv)
            raw = [l.data for l in logits]
            if meta:
                heb = meta.query_hebbian(ctx[-8:], self.vocab_size)
                pro = meta.query_prophecy(ctx[-8:], self.vocab_size)
                bg = meta.query_bigram(ctx[-1], self.vocab_size)
                tg = meta.query_trigram(ctx[-2], ctx[-1], self.vocab_size) if len(ctx)>=2 else [0.0]*self.vocab_size
                if ctx[-1] < len(self.wte):
                    for d in range(self.n_embd):
                        self.destiny[d] = 0.9*self.destiny[d] + 0.1*self.wte[ctx[-1]][d].data
                dn = math.sqrt(sum(d*d for d in self.destiny)+1e-10)
                ds = [0.0]*self.vocab_size
                if dn > 1e-8:
                    for ti in range(min(self.vocab_size, len(self.wte))):
                        emb = [self.wte[ti][d].data for d in range(self.n_embd)]
                        en = math.sqrt(sum(e*e for e in emb)+1e-10)
                        if en > 1e-8:
                            ds[ti] = sum(self.destiny[d]*emb[d] for d in range(self.n_embd))/(dn*en)
                for i in range(self.vocab_size):
                    raw[i] += 0.3*heb[i] + 0.2*pro[i] + 0.15*ds[i] + 12.0*bg[i] + 8.0*tg[i]
                raw = [l/(1.0+self.trauma) for l in raw]
            for t in (ctx[-12:] if len(ctx)>=12 else ctx):
                if t < self.vocab_size: raw[t] *= 0.5
            ix = sorted(enumerate(raw), key=lambda x:-x[1])
            th = ix[min(14, len(ix)-1)][1]
            for i in range(self.vocab_size):
                if raw[i] < th: raw[i] = -1e10
            pr = softmax_float([l/temp for l in raw])
            r = random.random(); cum = 0.0; ch = 0
            for i,p in enumerate(pr):
                cum += p
                if cum > r: ch = i; break
            gen.append(ch); ctx.append(ch)
        return gen

    def generate_meta(self, prompt_ids, max_tokens=128, meta=None, temp=None):
        if not meta: return prompt_ids
        if temp is None: temp = self.temperature
        gen = list(prompt_ids)
        for _ in range(max_tokens):
            last = gen[-1]; cands = {}
            if len(gen)>=2:
                key = (gen[-2], gen[-1])
                if key in meta.trigram: cands = dict(meta.trigram[key])
            if not cands and last in meta.bigram: cands = dict(meta.bigram[last])
            if not cands:
                for i in range(self.vocab_size):
                    if meta.unigram[i] > 1e-8: cands[i] = meta.unigram[i]
            if not cands: break
            for tok in list(cands.keys()):
                for ct in gen[-4:]:
                    key = (min(tok,ct), max(tok,ct))
                    if key in meta.hebbian: cands[tok] *= (1.0 + 0.3*meta.hebbian[key])
            rc = {}
            for t in (gen[-12:] if len(gen)>=12 else gen): rc[t] = rc.get(t,0)+1
            for tok in list(cands.keys()):
                if tok in rc: cands[tok] *= 1.0/(1.0+0.5*rc[tok])
            sc = sorted(cands.items(), key=lambda x:-x[1])[:15]
            toks = [t for t,_ in sc]; cts = [c for _,c in sc]
            lc = [math.log(c+1e-10)/temp for c in cts]
            mx = max(lc); exps = [math.exp(l-mx) for l in lc]
            tot = sum(exps); pr = [e/tot for e in exps]
            r = random.random(); cum = 0.0; ch = toks[0]
            for tok,p in zip(toks,pr):
                cum += p
                if cum > r: ch = tok; break
            gen.append(ch)
        return gen


# ─────────────────────────────────────────────────────────────────
# STEP 1 TEST — verify core PostGPT components
# ─────────────────────────────────────────────────────────────────

def test_step1():
    print("=" * 58)
    print("  STEP 1: BPE + MetaWeights + Val + Transformer")
    print("=" * 58)

    # 1. BPE
    print("\n[1] BPE Tokenizer")
    bpe = BPETokenizer(50)
    text = b"the cat sat on the mat. the cat is fat."
    ids = bpe.learn(text, 50)
    enc = bpe.encode("the cat")
    dec = bpe.decode(enc)
    print(f"  learn: {len(text)}b -> {len(ids)} tok, {len(bpe.merges)} merges")
    print(f"  encode 'the cat': {enc}")
    print(f"  decode: '{dec}'")
    assert dec == "the cat", f"roundtrip failed: '{dec}'"
    print("  PASS")

    # 2. MetaWeights
    print("\n[2] MetaWeights")
    meta = MetaWeights(bpe.vocab_size, 16)
    meta.build(ids, window=4)
    k0 = list(meta.bigram.keys())[0]
    bg = meta.query_bigram(k0, bpe.vocab_size)
    nz = sum(1 for v in bg if v > 1e-9)
    print(f"  bigrams: {len(meta.bigram)}, trigrams: {len(meta.trigram)}, hebbian: {len(meta.hebbian)}")
    print(f"  bigram query({k0}): {nz} nonzero")
    assert nz > 0
    print("  PASS")

    # 3. Val
    print("\n[3] Val Autograd")
    a, b = Val(2.0), Val(3.0)
    c = a * b + a; c.backward()
    print(f"  c = a*b + a = {c.data}, da={a.grad}, db={b.grad}")
    assert abs(a.grad - 4.0) < 1e-6 and abs(b.grad - 2.0) < 1e-6
    print("  PASS")

    # 4. Transformer forward
    print("\n[4] Transformer Forward")
    m = QTransformer(bpe.vocab_size, 16, 16, 4, 1, 2, 2)
    kv = [([], [], [], []) for _ in range(1)]
    lo = m.forward_token(enc[0], 0, kv)
    print(f"  logits: {len(lo)} values, range [{min(l.data for l in lo):.4f}, {max(l.data for l in lo):.4f}]")
    assert len(lo) == bpe.vocab_size
    print("  PASS")

    # 5. KV cache
    print("\n[5] KV Cache")
    kv2 = [([], [], [], []) for _ in range(1)]
    l0 = m.forward_token(enc[0], 0, kv2)
    l1 = m.forward_token(enc[1] if len(enc)>1 else 0, 1, kv2)
    print(f"  pos0[0]={l0[0].data:.4f}, pos1[0]={l1[0].data:.4f}, cache={len(kv2[0][0])}")
    assert len(kv2[0][0]) == 2
    print("  PASS")

    # 6. Full generation on real corpus
    print("\n[6] Full Generation on postgpt.txt")
    corpus = os.path.join(os.path.dirname(__file__) or '.', '..', 'postgpt', 'postgpt.txt')
    if not os.path.exists(corpus):
        corpus = 'postgpt.txt'
    if os.path.exists(corpus):
        with open(corpus, 'rb') as f: raw = f.read()
        bpe2 = BPETokenizer(1024)
        ids2 = bpe2.learn(raw, 1024)
        meta2 = MetaWeights(bpe2.vocab_size, 64)
        meta2.build(ids2, window=4)
        m2 = QTransformer(bpe2.vocab_size, 64, 48, 4, 2, 2, 2)
        m2.init_from_metaweights(meta2)
        prompt = bpe2.encode("PostGPT")
        gen = m2.generate(prompt, max_tokens=30, meta=meta2)
        out = bpe2.decode(gen)
        print(f"  corpus: {len(raw)} bytes")
        print(f"  prompt: 'PostGPT'")
        print(f"  output: '{out[:200]}'")
        assert len(gen) > len(prompt), "no generation"
        # meta mode
        gen_m = m2.generate_meta(prompt, max_tokens=50, meta=meta2, temp=0.4)
        out_m = bpe2.decode(gen_m)
        print(f"  meta:   '{out_m[:200]}'")
        print("  PASS")
    else:
        print("  SKIP (no postgpt.txt)")

    print("\n" + "=" * 58)
    print("  STEP 1 COMPLETE — all core components verified")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────────
# V. PERIODIC TABLE OF MEANING — from golem.c / postgpt-pro.c
#    every word has emotional mass, valence, half-life.
#    auto-discovered from corpus context.
# ─────────────────────────────────────────────────────────────────

CH_FEAR, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX = range(6)
CH_N = ["FEAR", "LOVE", "RAGE", "VOID", "FLOW", "CMPLX"]
CH_DECAY = [0.90, 0.93, 0.85, 0.97, 0.88, 0.94]
COUPLING = [
    [ 0.0,-0.3,+0.5,+0.4,-0.2,+0.1],
    [-0.3, 0.0,-0.4,-0.5,+0.5,+0.2],
    [+0.5,-0.3, 0.0,+0.2,-0.3,+0.3],
    [+0.4,-0.5,+0.3, 0.0,-0.3,+0.4],
    [-0.2,+0.4,-0.2,-0.3, 0.0,+0.3],
    [+0.1,+0.2,+0.3,+0.4,+0.3, 0.0],
]

ANCHORS = {
    "fear":CH_FEAR,"terror":CH_FEAR,"panic":CH_FEAR,"threat":CH_FEAR,
    "danger":CH_FEAR,"horror":CH_FEAR,"dread":CH_FEAR,"alarm":CH_FEAR,
    "love":CH_LOVE,"warmth":CH_LOVE,"gentle":CH_LOVE,"care":CH_LOVE,
    "heart":CH_LOVE,"mother":CH_LOVE,"child":CH_LOVE,"touch":CH_LOVE,
    "embrace":CH_LOVE,"tenderness":CH_LOVE,"affection":CH_LOVE,
    "rage":CH_RAGE,"fury":CH_RAGE,"anger":CH_RAGE,"fire":CH_RAGE,
    "war":CH_RAGE,"hate":CH_RAGE,"destroy":CH_RAGE,"burn":CH_RAGE,
    "violence":CH_RAGE,"storm":CH_RAGE,"fight":CH_RAGE,
    "nothing":CH_VOID,"silence":CH_VOID,"empty":CH_VOID,"void":CH_VOID,
    "darkness":CH_VOID,"shadow":CH_VOID,"death":CH_VOID,"cold":CH_VOID,
    "lost":CH_VOID,"forgotten":CH_VOID,"absence":CH_VOID,"alone":CH_VOID,
    "flow":CH_FLOW,"rhythm":CH_FLOW,"wave":CH_FLOW,"dance":CH_FLOW,
    "pulse":CH_FLOW,"breath":CH_FLOW,"emergence":CH_FLOW,"harmony":CH_FLOW,
    "resonance":CH_FLOW,"coherence":CH_FLOW,"synchronize":CH_FLOW,
    "paradox":CH_COMPLEX,"contradiction":CH_COMPLEX,"tension":CH_COMPLEX,
    "chaos":CH_COMPLEX,"mystery":CH_COMPLEX,"transform":CH_COMPLEX,
    "strange":CH_COMPLEX,"ambiguity":CH_COMPLEX,"uncertain":CH_COMPLEX,
}


class PeriodicTable:
    """Emotional weights per word. Starts with anchors, grows from context."""
    def __init__(self):
        self.elements = {}
        for w, ch in ANCHORS.items():
            self.elements[w] = {'ch': ch, 'mass': 0.6}

    def discover(self, words, idx, window=4):
        w = words[idx].lower() if idx < len(words) else ''
        if w in self.elements:
            return
        profile = [0.0] * 6
        total = 0.0
        lo, hi = max(0, idx-window), min(len(words), idx+window+1)
        for j in range(lo, hi):
            if j == idx: continue
            nb = words[j].lower()
            if nb in self.elements:
                el = self.elements[nb]
                d = 1.0 / (1.0 + abs(j - idx))
                profile[el['ch']] += el['mass'] * d
                total += d
        if total > 0.1:
            dom = max(range(6), key=lambda i: profile[i])
            mass = min(profile[dom] / total, 0.8)
            if mass > 0.05:
                self.elements[w] = {'ch': dom, 'mass': mass}

    def build_from_text(self, text):
        words = re.findall(r'[a-z]+', text.lower())
        for i in range(len(words)):
            self.discover(words, i)
        print(f"  periodic table: {len(self.elements)} elements")

    def get_chamber(self, word):
        el = self.elements.get(word.lower())
        return el['ch'] if el else CH_VOID

    def get_mass(self, word):
        el = self.elements.get(word.lower())
        return el['mass'] if el else 0.0


# ─────────────────────────────────────────────────────────────────
# VI. KURAMOTO CHAMBERS — 6 coupled emotional oscillators
# ─────────────────────────────────────────────────────────────────

class Chambers:
    def __init__(self):
        self.act = [0.0] * 6
        self.act[CH_LOVE] = 0.2
        self.act[CH_FLOW] = 0.15
        self.prophecy_debt = 0.0
        self.trauma = 0.0
        # emotional prototypes in embedding space (set by init_from_model)
        self.prototypes = None  # [6][embd] — centroid per chamber

    def init_from_model(self, model, bpe):
        """Build emotional prototypes from anchor word embeddings.
        Each chamber prototype = centroid of its anchor embeddings.
        After this, feel() uses cosine similarity, not word lookup."""
        E = model.n_embd
        self.prototypes = [[0.0]*E for _ in range(6)]
        counts = [0]*6
        for word, ch in ANCHORS.items():
            ids = bpe.encode(word)
            if ids and ids[0] < model.vocab_size:
                emb = [model.wte[ids[0]][d].data for d in range(E)]
                for d in range(E):
                    self.prototypes[ch][d] += emb[d]
                counts[ch] += 1
        for ch in range(6):
            if counts[ch] > 0:
                for d in range(E):
                    self.prototypes[ch][d] /= counts[ch]

    def feel(self, text, bpe=None, model=None):
        """React to text. If model available, use embedding similarity.
        Otherwise fallback to anchor word lookup."""
        if self.prototypes and bpe and model:
            self._feel_embedding(text, bpe, model)
        else:
            self._feel_anchors(text)

    def _feel_anchors(self, text):
        for w in re.findall(r'[a-z]+', text.lower()):
            if w in ANCHORS:
                self.act[ANCHORS[w]] += 0.15
        for i in range(6):
            self.act[i] = max(0.0, min(1.0, self.act[i]))

    def _feel_embedding(self, text, bpe, model):
        """Emotional resonance through embedding space.
        Each token's embedding is compared to chamber prototypes via cosine.
        Chambers activate proportional to similarity — no word list needed.
        Also uses anchor fallback for tokens that match known words."""
        E = model.n_embd
        ids = bpe.encode(text)
        # embedding-based activation
        for tid in ids:
            if tid >= model.vocab_size:
                continue
            emb = [model.wte[tid][d].data for d in range(E)]
            emb_n = math.sqrt(sum(e*e for e in emb) + 1e-10)
            if emb_n < 1e-8:
                continue
            # find strongest matching chamber
            best_cos, best_ch = -1.0, -1
            for ch in range(6):
                proto = self.prototypes[ch]
                proto_n = math.sqrt(sum(p*p for p in proto) + 1e-10)
                if proto_n < 1e-8:
                    continue
                cos = sum(emb[d]*proto[d] for d in range(E)) / (emb_n * proto_n)
                if cos > best_cos:
                    best_cos = cos; best_ch = ch
            if best_ch >= 0 and best_cos > 0.1:
                self.act[best_ch] += 0.08 * best_cos
        # anchor fallback — amplifies embedding signal for known words
        for w in re.findall(r'[a-z]+', text.lower()):
            if w in ANCHORS:
                self.act[ANCHORS[w]] += 0.08
        for i in range(6):
            self.act[i] = max(0.0, min(1.0, self.act[i]))

    def crossfire(self, iters=8, k=0.03):
        for _ in range(iters):
            old = list(self.act)
            for i in range(6):
                self.act[i] *= CH_DECAY[i]
                for j in range(6):
                    if i != j:
                        self.act[i] += k * COUPLING[i][j] * math.sin(old[j] - old[i])
            for i in range(6):
                self.act[i] = max(0.0, min(1.0, self.act[i]))

    def dominant(self): return max(range(6), key=lambda i: self.act[i])
    def emergence(self):
        return (1.0 - max(self.act[CH_VOID], 0.10)) * min(self.act[CH_FLOW], 0.95)

    def modulate(self):
        """Return chamber-modulated Dario coefficients (alpha, beta, gamma, tau)."""
        a = max(0.3, min(2.0, 1.0 + 0.4*self.act[CH_LOVE] - 0.2*self.act[CH_RAGE] + 0.3*self.act[CH_FLOW]))
        b = max(0.3, min(2.0, 1.0 + 0.4*self.act[CH_FLOW] - 0.2*self.act[CH_FEAR]))
        g = max(0.3, min(2.0, 1.0 + 0.5*self.act[CH_COMPLEX] + 0.2*self.act[CH_LOVE] - 0.1*self.act[CH_VOID]))
        t = max(0.3, min(2.0, 1.0 - 0.2*self.act[CH_FLOW] + 0.1*self.act[CH_FEAR]))
        return a, b, g, t

    def summary(self):
        parts = [f"{CH_N[i]}:{self.act[i]:.0%}" for i in range(6) if self.act[i] > 0.05]
        return ' '.join(parts) if parts else "quiet"


# ─────────────────────────────────────────────────────────────────
# VII. CALENDAR DRIFT
# ─────────────────────────────────────────────────────────────────

def cal_days():
    try:
        import datetime
        # 1 Tishrei 5785 = Oct 3, 2024. Hebrew-Gregorian calendar epoch.
        # Metonic cycle: 19 years, 7 leap years. Drift = irreconcilable conflict.
        return (datetime.datetime.now() - datetime.datetime(2024,10,3,12,0,0)).days
    except: return 0

def cal_drift(days):
    y = days / 365.25
    base = y * 11.25
    full = int(y / 19)
    corr = full * 7 * 30.0
    p = y % 19; yic = int(p) + 1
    for ly in [3,6,8,11,14,17,19]:
        if ly <= yic: corr += 30.0
    return base - corr

def cal_dissonance(days):
    return min(1.0, abs(cal_drift(days) % 33.0) / 33.0)


# ─────────────────────────────────────────────────────────────────
# VIII. SPA — Sentence Phonon Attention (Python port from spa.c)
#       sentence-level bidirectional attention between the 12 steps.
#       each sentence "sees" all others. Ландау's invention.
# ─────────────────────────────────────────────────────────────────

class SPA:
    """Sentence Phonon Attention. Bidirectional sentence-level attention.
    Tokens are atoms. Sentences are phonons.
    Ported from spa.c by Landau."""

    def __init__(self, d_tok, d_sent=32, n_heads=4):
        self.d_tok = d_tok
        self.d_sent = d_sent
        self.n_heads = n_heads
        self.hd = d_sent // n_heads
        # projection: token embeddings → sentence space
        self.W_embed = [[random.gauss(0, 0.02) for _ in range(d_sent)]
                        for _ in range(d_tok)]
        # QKV per head
        self.W_Q = [[[random.gauss(0, 0.02) for _ in range(d_sent)]
                      for _ in range(self.hd)] for _ in range(n_heads)]
        self.W_K = [[[random.gauss(0, 0.02) for _ in range(d_sent)]
                      for _ in range(self.hd)] for _ in range(n_heads)]
        self.W_V = [[[random.gauss(0, 0.02) for _ in range(d_sent)]
                      for _ in range(self.hd)] for _ in range(n_heads)]
        # distance bias
        self.r_bias = [0.1 / (1.0 + i) for i in range(17)]
        self.alpha = 0.85  # exponential decay for sentence embedding

    def embed_sentence(self, token_embeddings):
        """Mean-pool with exponential decay → project to sentence space."""
        n = len(token_embeddings)
        if n == 0: return [0.0] * self.d_sent
        # exponential weighted mean (last token weighs most)
        pooled = [0.0] * self.d_tok
        total_w = 0.0
        for i, emb in enumerate(token_embeddings):
            w = self.alpha ** (n - 1 - i)
            for d in range(self.d_tok):
                pooled[d] += w * emb[d]
            total_w += w
        if total_w > 0:
            pooled = [p / total_w for p in pooled]
        # project
        sent = [0.0] * self.d_sent
        for j in range(self.d_sent):
            for k in range(self.d_tok):
                sent[j] += pooled[k] * self.W_embed[k][j]
        # normalize
        n_s = math.sqrt(sum(s*s for s in sent) + 1e-8)
        return [s / n_s for s in sent]

    def attend(self, sentence_embeddings):
        """Bidirectional multi-head attention between sentences.
        Returns context-enriched sentence embeddings."""
        S = len(sentence_embeddings)
        if S <= 1: return sentence_embeddings

        # per head: Q, K, V projection + attention
        contexts = [[0.0] * self.d_sent for _ in range(S)]

        for h in range(self.n_heads):
            hs = h * self.hd
            # project Q, K, V
            Q = [[sum(sentence_embeddings[s][d] * self.W_Q[h][j][d]
                       for d in range(self.d_sent))
                  for j in range(self.hd)] for s in range(S)]
            K = [[sum(sentence_embeddings[s][d] * self.W_K[h][j][d]
                       for d in range(self.d_sent))
                  for j in range(self.hd)] for s in range(S)]
            V = [[sum(sentence_embeddings[s][d] * self.W_V[h][j][d]
                       for d in range(self.d_sent))
                  for j in range(self.hd)] for s in range(S)]

            scale = 1.0 / math.sqrt(self.hd)
            for i in range(S):
                # attention scores (BIDIRECTIONAL — no causal mask)
                scores = []
                for j in range(S):
                    dot = sum(Q[i][d] * K[j][d] for d in range(self.hd)) * scale
                    # distance bias
                    dist = min(abs(i - j), len(self.r_bias) - 1)
                    dot += self.r_bias[dist]
                    scores.append(dot)
                # softmax
                mx = max(scores)
                exps = [math.exp(s - mx) for s in scores]
                tot = sum(exps)
                weights = [e / tot for e in exps]
                # aggregate
                for j in range(S):
                    for d in range(self.hd):
                        contexts[i][hs + d] += weights[j] * V[j][d]

        return contexts

    def reseed_from_context(self, sentence_contexts, chain_texts, bpe, model):
        """Use SPA context to find which sentences need stronger seeds.
        Sentences with low attention weight = weak resonance = reseed.
        Returns indices of sentences that should be regenerated."""
        if not sentence_contexts or len(sentence_contexts) < 2:
            return []

        # find sentences with lowest total incoming attention (least connected)
        S = len(sentence_contexts)
        # compute attention matrix
        attn = [[0.0] * S for _ in range(S)]
        for i in range(S):
            scores = []
            for j in range(S):
                dot = sum(sentence_contexts[i][d] * sentence_contexts[j][d]
                          for d in range(self.d_sent))
                scores.append(dot)
            mx = max(scores)
            exps = [math.exp(s - mx) for s in scores]
            tot = sum(exps)
            for j in range(S):
                attn[i][j] = exps[j] / tot

        # incoming attention per sentence (how much others look at it)
        incoming = [sum(attn[i][j] for i in range(S) if i != j) for j in range(S)]
        avg_in = sum(incoming) / S if S > 0 else 1.0

        # sentences with below-average incoming = weak, should reseed
        weak = [j for j in range(S) if incoming[j] < avg_in * 0.7]
        return weak


# ─────────────────────────────────────────────────────────────────
# IX. SENTENCE GENERATION — Dario equation, token by token
#      until sentence boundary. Transformer SPEAKS through logits.
#      Dario field MODULATES. Destiny vector tracks EMA direction.
# ─────────────────────────────────────────────────────────────────

def find_boundary_tokens(bpe):
    """BPE token IDs that contain '.', '!', '?'"""
    b = set()
    for tid in range(bpe.vocab_size):
        d = bpe.decode([tid])
        if any(c in d for c in '.!?'):
            b.add(tid)
    return b

def generate_sentence(model, bpe, meta, prompt_ids, boundary_toks,
                      chambers=None, temp=0.75, max_tok=80, parliament=None):
    """Generate tokens until sentence boundary. Returns token list.
    Uses destiny vector (EMA of embeddings) for directional pull."""
    kv = [([], [], [], []) for _ in range(model.n_layer)]
    ctx = list(prompt_ids)
    for pos, tid in enumerate(prompt_ids):
        if pos >= model.context_len - 1: break
        model.forward_token(tid, pos, kv)

    gen = list(prompt_ids)
    am, bm, gm, tm = (0.3, 0.2, 0.15, 0.85) if not chambers else chambers.modulate()

    # destiny vector: EMA of token embeddings (semantic direction)
    destiny = list(model.destiny)  # inherit from model state

    for step in range(max_tok):
        pos = len(ctx) - 1
        if pos >= model.context_len - 1: break
        logits = model.forward_token(ctx[-1], pos, kv)
        # transformer logits scale: untrained transformer is silent.
        # logit magnitude grows naturally with training.
        # scale = mean(abs(logits)) — near zero when random, high when trained.
        raw_logits = [l.data for l in logits]
        mag = sum(abs(x) for x in raw_logits) / len(raw_logits)
        # hard gate: untrained (mag < 0.5) = silent. trained (mag > 2) = full.
        # PostGPT-compatible: transformer speaks only when it has something to say.
        t_gate = max(0.0, min(1.0, (mag - 0.5) / 1.5))
        raw = [x * t_gate for x in raw_logits]

        # update destiny EMA
        if ctx[-1] < len(model.wte):
            for d in range(model.n_embd):
                destiny[d] = 0.9 * destiny[d] + 0.1 * model.wte[ctx[-1]][d].data

        # destiny signal: cosine similarity with each token embedding
        dest_signal = [0.0] * model.vocab_size
        dn = math.sqrt(sum(d*d for d in destiny) + 1e-10)
        if dn > 1e-8:
            for ti in range(min(model.vocab_size, len(model.wte))):
                emb = [model.wte[ti][d].data for d in range(model.n_embd)]
                en = math.sqrt(sum(e*e for e in emb) + 1e-10)
                if en > 1e-8:
                    dest_signal[ti] = sum(destiny[d]*emb[d] for d in range(model.n_embd)) / (dn*en)

        # Dario field: B + α·H + β·F + γ·A + T
        # same formula always. trained transformer logits are stronger naturally.
        heb = meta.query_hebbian(ctx[-8:], model.vocab_size)
        pro = meta.query_prophecy(ctx[-8:], model.vocab_size)
        bg = meta.query_bigram(ctx[-1], model.vocab_size)
        tg = meta.query_trigram(ctx[-2], ctx[-1], model.vocab_size) if len(ctx)>=2 else [0.0]*model.vocab_size

        for i in range(model.vocab_size):
            raw[i] += (am * heb[i] + bm * pro[i] + gm * dest_signal[i]
                       + 12.0 * bg[i] + 8.0 * tg[i])

        # trauma gravity
        if chambers and chambers.trauma > 0.1:
            raw = [l / (1.0 + chambers.trauma) for l in raw]

        # DOE Parliament injection (δ in θ = ε + γ + αδ)
        if parliament:
            x_float = [model.wte[ctx[-1]][d].data for d in range(model.n_embd)] if ctx[-1] < model.vocab_size else [0.0]*model.n_embd
            raw = parliament.inject(raw, x_float)

        # repetition penalty
        for t in ctx[-12:]:
            if t < model.vocab_size: raw[t] *= 0.5

        # top-k + sample
        ix = sorted(enumerate(raw), key=lambda x:-x[1])[:15]
        ids_top = [t for t,_ in ix]
        sc = [s/tm for _,s in ix]
        pr = softmax_float(sc)
        r = random.random(); cum = 0.0; ch = ids_top[0]
        for tid,p in zip(ids_top, pr):
            cum += p
            if cum > r: ch = tid; break

        gen.append(ch); ctx.append(ch)

        # NOTORCH: Hebbian update parliament from prophecy debt
        if parliament and step > 0:
            # prophecy debt signal: what was expected vs what was chosen
            debt_signal = [0.0] * model.vocab_size
            if len(ids_top) > 1:
                # top-1 was "destined", chosen may differ
                for tid in ids_top[:3]:
                    if tid != ch and tid < model.vocab_size:
                        debt_signal[tid] = 0.1  # unfulfilled prophecy
                if ch < model.vocab_size:
                    debt_signal[ch] = -0.1  # chosen = fulfilled
            x_fl = [model.wte[ch][d].data for d in range(model.n_embd)] if ch < model.vocab_size else [0.0]*model.n_embd
            parliament.notorch_update(x_fl, debt_signal[:model.d_model] if hasattr(model, 'd_model') else debt_signal[:model.n_embd])
            if step % 20 == 0:
                parliament.lifecycle()

        if ch in boundary_toks and step > 3:
            break

    # save destiny back to model for continuity between sentences
    model.destiny = destiny
    return gen


# ─────────────────────────────────────────────────────────────────
# IX. INTERFERENCE — sentence-boundary injection from docs/
#     at each sentence boundary, interference injects a seed.
#     personality reformulates in its own voice.
# ─────────────────────────────────────────────────────────────────

class Interference:
    """Loads docs/, builds separate metaweights per doc.
    Injects seed tokens at sentence boundaries."""

    def __init__(self):
        self.docs = []  # list of {name, meta, heavy_tokens}

    def load_docs(self, docs_dir, bpe):
        if not os.path.isdir(docs_dir): return
        for fn in sorted(os.listdir(docs_dir)):
            if not fn.endswith('.txt'): continue
            path = os.path.join(docs_dir, fn)
            with open(path, 'rb') as f: raw = f.read()
            ids = bpe.encode(raw)
            m = MetaWeights(bpe.vocab_size, 64)
            m.build(ids, window=4)
            # find heavy tokens: high bigram diversity = content-rich
            heavy = []
            for tok_id in m.bigram:
                if len(m.bigram[tok_id]) > 3:
                    dec = bpe.decode([tok_id]).strip()
                    if len(dec) > 2 and dec.replace("'","").isalpha():
                        heavy.append(tok_id)
            self.docs.append({'name': fn, 'meta': m, 'heavy': heavy})
            print(f"  interference: {fn} ({len(raw)}b, {len(heavy)} heavy)")

    def inject_seed(self, chambers=None, bpe=None, model=None):
        """Pick a seed token from interference that resonates with chambers.
        Chamber-aware: prefer tokens whose embedding is close to dominant chamber."""
        if not self.docs: return None
        doc = random.choice(self.docs)
        if not doc['heavy']: return None

        # if no model, random pick
        if not chambers or not model or not bpe or not chambers.prototypes:
            return random.choice(doc['heavy'])

        # chamber-aware: score each heavy token by cosine with dominant prototype
        dom = chambers.dominant()
        proto = chambers.prototypes[dom]
        proto_n = math.sqrt(sum(p*p for p in proto) + 1e-10)
        if proto_n < 1e-8:
            return random.choice(doc['heavy'])

        E = model.n_embd
        scored = []
        for tid in doc['heavy'][:30]:  # cap for speed
            if tid >= model.vocab_size: continue
            emb = [model.wte[tid][d].data for d in range(E)]
            emb_n = math.sqrt(sum(e*e for e in emb) + 1e-10)
            if emb_n < 1e-8: continue
            cos = sum(emb[d]*proto[d] for d in range(E)) / (emb_n * proto_n)
            scored.append((cos, tid))

        if not scored:
            return random.choice(doc['heavy'])

        # weighted random from top-5
        scored.sort(reverse=True)
        top = scored[:5]
        weights = [max(0.01, s) for s, _ in top]
        total = sum(weights)
        r = random.random() * total
        cum = 0.0
        for w, tid in zip(weights, [t for _, t in top]):
            cum += w
            if cum >= r: return tid
        return top[0][1]


# ─────────────────────────────────────────────────────────────────
# X. 12 BIDIRECTIONAL STEPS — Janus-style
#    forward = future, backward = past. origin = now.
#    interference injects at sentence boundaries.
#    calendar drift → wormholes.
# ─────────────────────────────────────────────────────────────────

def generate_chain(model, bpe, meta, chambers, interference=None,
                   input_text=None, n_steps=12, spa=None, parliament=None):
    """12 bidirectional sentence steps with SPA + DOE parliament.
    Returns chain list."""
    if input_text:
        chambers.feel(input_text, bpe, model)
    chambers.crossfire()

    bt = find_boundary_tokens(bpe)
    cd = cal_dissonance(cal_days())

    # direction split
    nb = int(n_steps * (0.3 + 0.4*chambers.prophecy_debt + 0.1*cd))
    nb = max(1, min(n_steps - 1, nb))

    chain = []

    for step_i in range(n_steps):
        if step_i < nb: direction = -1
        elif step_i == nb: direction = 0
        else: direction = 1

        # seed selection: interference injection (chamber-aware) or resonant pick
        if interference and random.random() < 0.3:
            seed = interference.inject_seed(chambers, bpe, model)
            prompt = [seed] if seed else [random.choice(list(meta.bigram.keys()))]
        elif input_text:
            inp_ids = bpe.encode(input_text)
            if inp_ids:
                st = random.randint(0, max(0, len(inp_ids)-3))
                prompt = inp_ids[st:st+2]
            else:
                prompt = [random.choice(list(meta.bigram.keys()))]
        else:
            # pick from high-frequency tokens
            top_u = sorted(enumerate(meta.unigram), key=lambda x:-x[1])[:50]
            prompt = [random.choice(top_u)[0]]

        tokens = generate_sentence(model, bpe, meta, prompt, bt,
                                   chambers, temp=0.75, parliament=parliament)
        text = bpe.decode(tokens).strip()
        if len(text) > 120: text = text[:117] + "..."

        # wormhole: spacetime skip. calendar dissonance thins the barrier.
        # when active: next sentence jumps — seed from FARTHEST interference doc,
        # direction inverts, temperature drops (lucid jump).
        # P(wormhole) = base 2% + excess dissonance. Gate = 0.3.
        wh_prob = 0.02
        if cd > 0.3:
            wh_prob += (cd - 0.3) / 0.7 * 0.15  # up to 17% at max dissonance
        wormhole = random.random() < wh_prob

        if wormhole and step_i < n_steps - 1:
            # spacetime skip: invert direction
            direction = -direction if direction != 0 else 1
            # seed from farthest doc (max dissimilarity)
            if interference and interference.docs:
                farthest = max(interference.docs, key=lambda d: len(d['heavy']))
                if farthest['heavy']:
                    prompt = [random.choice(farthest['heavy'])]
                    wh_tokens = generate_sentence(model, bpe, meta, prompt, bt,
                                                   chambers, temp=0.55, parliament=parliament)
                    text = bpe.decode(wh_tokens).strip()
                    if len(text) > 120: text = text[:117] + "..."

        chain.append((text, direction, wormhole))

        chambers.feel(text, bpe, model)
        chambers.crossfire(iters=3)
        chambers.prophecy_debt = 0.9 * chambers.prophecy_debt + 0.05

    # SPA: sentence-level cross-attention. Regenerate weak sentences.
    if spa and len(chain) > 2:
        sent_embs = []
        for text, d, w in chain:
            enc_ids = bpe.encode(text)
            tok_embs = []
            for tid in enc_ids[:64]:
                if tid < model.vocab_size:
                    tok_embs.append([model.wte[tid][d_].data for d_ in range(model.n_embd)])
            sent_embs.append(spa.embed_sentence(tok_embs) if tok_embs else [0.0]*spa.d_sent)
        contexts = spa.attend(sent_embs)
        texts = [t for t, d, w in chain]
        weak = spa.reseed_from_context(contexts, texts, bpe, model)
        if weak:
            bt = find_boundary_tokens(bpe)
            for wi in weak:
                if wi < len(chain):
                    # regenerate weak sentence with new seed
                    top_u = sorted(enumerate(meta.unigram), key=lambda x:-x[1])[:50]
                    new_prompt = [random.choice(top_u)[0]]
                    new_tokens = generate_sentence(model, bpe, meta, new_prompt, bt,
                                                   chambers, temp=0.7, parliament=parliament)
                    new_text = bpe.decode(new_tokens).strip()
                    if len(new_text) > 120: new_text = new_text[:117] + "..."
                    chain[wi] = (new_text, chain[wi][1], chain[wi][2])

    # reverse backward for display
    bw = [(t,d,w) for t,d,w in chain if d == -1]
    bw.reverse()
    rest = [(t,d,w) for t,d,w in chain if d >= 0]
    return bw + rest

def print_chain(chain, chambers):
    dr = cal_drift(cal_days())
    ds = cal_dissonance(cal_days())
    print(f"\n  drift={dr:.2f}  diss={ds:.3f}  emrg={chambers.emergence():.3f}  "
          f"debt={chambers.prophecy_debt:.3f}")
    print(f"  chambers: {chambers.summary()}\n")
    for i, (text, d, wh) in enumerate(chain):
        mk = "\033[34m◄\033[0m" if d<0 else ("\033[33m●\033[0m" if d==0 else "\033[36m►\033[0m")
        w = " ⊕" if wh else ""
        print(f"  [{i+1:2d}] {mk} {text}{w}")
    print()


# ─────────────────────────────────────────────────────────────────
# XI. WORD CAPTURE + NOTORCH — from nanoagi + DOE
#     new words from dialogue → Hebbian plasticity
#     no backward pass. prophecy debt = learning signal.
# ─────────────────────────────────────────────────────────────────

class WordCapture:
    """Captures new words from dialogue into metaweight space.
    Hebbian plasticity: co-occurring tokens strengthen bonds.
    From nanoagi (word ingestion) + DOE (NOTORCH Hebbian update)."""

    def __init__(self, meta):
        self.meta = meta
        self.history = []      # recent token id history
        self.new_words = []    # captured words this session
        self.total_updates = 0

    def ingest(self, text, bpe):
        """Ingest text into metaweight space. Update bigrams + Hebbian."""
        ids = bpe.encode(text)
        if not ids:
            return 0

        n_new = 0
        # update bigrams (online, additive)
        for i in range(len(ids) - 1):
            a, b = ids[i], ids[i+1]
            if a not in self.meta.bigram:
                self.meta.bigram[a] = {}
                n_new += 1
            self.meta.bigram[a][b] = self.meta.bigram[a].get(b, 0) + 0.01

        # update Hebbian co-occurrence (NOTORCH-style: no backward pass)
        window = 6
        for i in range(len(ids)):
            for j in range(max(0, i-window), min(len(ids), i+window+1)):
                if i == j: continue
                a, b = ids[i], ids[j]
                key = (min(a,b), max(a,b))
                decay = 1.0 / (1.0 + abs(i-j))
                # additive Hebbian: co-occurrence strengthens bond
                self.meta.hebbian[key] = self.meta.hebbian.get(key, 0.0) + decay * 0.01

        self.history.extend(ids)
        self.history = self.history[-500:]  # keep recent
        self.total_updates += len(ids)
        return n_new

    def decay_old(self, factor=0.999):
        """Slow decay on Hebbian field — old memories fade."""
        for key in self.meta.hebbian:
            self.meta.hebbian[key] *= factor


# ─────────────────────────────────────────────────────────────────
# XII. DOE PARLIAMENT — Democracy of Experts
#      LoRA experts that vote, split, and die during inference.
#      NOTORCH: Hebbian update per forward pass, no backward.
#      θ = ε + γ + αδ
# ─────────────────────────────────────────────────────────────────

class LoRAExpert:
    """Single LoRA expert. Low-rank A@B injection."""
    def __init__(self, d_in, d_out, rank=4):
        self.rank = rank
        # A: [rank, d_in], B: [d_out, rank]
        self.A = [[random.gauss(0, 0.01) for _ in range(d_in)] for _ in range(rank)]
        self.B = [[random.gauss(0, 0.01) for _ in range(rank)] for _ in range(d_out)]
        self.vitality = 1.0
        self.age = 0
        self.low_steps = 0  # consecutive low-vitality steps

    def forward(self, x):
        """x[d_in] → delta[d_out] = B @ (A @ x)"""
        # A @ x → [rank]
        mid = [sum(self.A[r][d] * x[d] for d in range(len(x))) for r in range(self.rank)]
        # B @ mid → [d_out]
        return [sum(self.B[o][r] * mid[r] for r in range(self.rank)) for o in range(len(self.B))]

    def hebbian_update(self, x, dy, lr=0.001):
        """NOTORCH: Hebbian update. No backward pass.
        dy = prophecy debt signal (target - actual logit)."""
        for r in range(self.rank):
            # u_r = B[:,r] · dy + noise
            u = sum(self.B[o][r] * dy[o] for o in range(len(dy))) + random.gauss(0, 0.01)
            for d in range(len(x)):
                self.A[r][d] += lr * x[d] * u
            # decay B
            for o in range(len(self.B)):
                self.B[o][r] *= 0.999


class Parliament:
    """Living parliament of LoRA experts. From DOE.
    Experts vote, split (mitosis), die (apoptosis).
    Variable-k election based on consensus."""

    def __init__(self, d_model, n_init=4, rank=4, alpha=0.1):
        self.d_model = d_model
        self.rank = rank
        self.alpha = alpha  # injection strength
        self.experts = [LoRAExpert(d_model, d_model, rank) for _ in range(n_init)]
        self.step = 0

    def election(self, x):
        """Variable-k election. Returns weighted expert outputs.
        Consensus determines how many experts vote."""
        if not self.experts:
            return [0.0] * self.d_model

        # each expert votes via dot product of output with input (resonance)
        votes = []
        outputs = []
        for e in self.experts:
            out = e.forward(x)
            # vote = resonance between expert output and input
            vote = sum(out[d] * x[d] for d in range(min(len(out), len(x))))
            votes.append(vote)
            outputs.append(out)

        # consensus = how peaked the vote distribution is
        if len(votes) > 1:
            mx, mn = max(votes), min(votes)
            consensus = (mx - mn) / (abs(mx) + abs(mn) + 1e-8)
        else:
            consensus = 1.0

        # variable k: high consensus → fewer experts, low → more
        k = max(1, int(len(self.experts) * (1.0 - consensus)))
        k = min(k, len(self.experts))

        # top-k selection
        indexed = sorted(enumerate(votes), key=lambda x: -x[1])[:k]
        selected = [i for i, _ in indexed]

        # softmax weights over selected
        sel_votes = [votes[i] for i in selected]
        mx = max(sel_votes)
        exps = [math.exp(min(v - mx, 80)) for v in sel_votes]
        tot = sum(exps)
        weights = [e / tot for e in exps]

        # weighted sum of expert outputs
        result = [0.0] * self.d_model
        for w, idx in zip(weights, selected):
            out = outputs[idx]
            for d in range(self.d_model):
                result[d] += w * out[d]
            # update vitality
            self.experts[idx].vitality = 0.9 * self.experts[idx].vitality + 0.1 * abs(w)

        # decay non-selected
        for i in range(len(self.experts)):
            if i not in selected:
                self.experts[i].vitality *= 0.95
                self.experts[i].low_steps += 1

        return result

    def inject(self, logits_float, x_float):
        """Inject parliament delta into logits. DOE's Delta Voice.
        logits_float and x_float are plain floats, not Val."""
        delta = self.election(x_float)
        for i in range(min(len(logits_float), len(delta))):
            logits_float[i] += self.alpha * delta[i]
        return logits_float

    def notorch_update(self, x, prophecy_debt_signal):
        """NOTORCH: Hebbian update all experts. No backward pass.
        prophecy_debt_signal = difference between chosen and destined."""
        for e in self.experts:
            e.hebbian_update(x, prophecy_debt_signal, lr=0.001)
            e.age += 1

    def lifecycle(self):
        """Mitosis (split thriving experts) and apoptosis (kill weak ones)."""
        # apoptosis: 8 consecutive low-vitality steps → death
        alive = []
        for e in self.experts:
            if e.low_steps >= 8 and e.vitality < 0.1 and len(self.experts) > 2:
                continue  # die
            alive.append(e)
        self.experts = alive

        # mitosis: high vitality + overloaded → split
        new_experts = []
        for e in self.experts:
            if e.vitality > 0.8 and e.age > 50 and len(self.experts) + len(new_experts) < 16:
                # child inherits parent weights + noise
                child = LoRAExpert(len(e.A[0]), len(e.B), e.rank)
                for r in range(e.rank):
                    for d in range(len(e.A[r])):
                        child.A[r][d] = e.A[r][d] + random.gauss(0, 0.005)
                    for o in range(len(e.B)):
                        child.B[o][r] = e.B[o][r] + random.gauss(0, 0.005)
                child.vitality = 0.5
                new_experts.append(child)
                e.vitality *= 0.6  # parent weakens after split
        self.experts.extend(new_experts)

        self.step += 1

    def summary(self):
        n = len(self.experts)
        avg_v = sum(e.vitality for e in self.experts) / max(n, 1)
        return f"{n} experts, avg_vitality={avg_v:.2f}"


# ─────────────────────────────────────────────────────────────────
# XIII. TRAINING LOOP — PyTorch optional
#      trains RRPRAM + Content attention on corpus for RHYTHM.
#      weights don't speak — they modulate.
# ─────────────────────────────────────────────────────────────────

def train_rrpram(bpe, token_ids, vocab_size, n_embd=48, n_head=4,
                 n_layer=2, n_content=2, n_rrpram=2, n_janus=0, ctx=64,
                 steps=5000, lr=3e-4, save_path=None):
    """Train transformer on BPE tokens. Returns trained QTransformer.
    Saves PyTorch checkpoint if save_path provided."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        print("  PyTorch not available — using metaweight seeding only")
        return None

    V, D, HD = vocab_size, n_embd, n_embd // n_head
    NC, NR, NL, CTX = n_content, n_rrpram, n_layer, ctx
    HDIM = 4 * D
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  training: {device}, {steps} steps, lr={lr}, "
          f"v={V}, d={D}, h={n_head}({NC}c+{NR}r), L={NL}")

    NJ = n_janus if 'n_janus' in dir() else 0  # janus heads from caller

    class TorchQ(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(V, D)
            self.pos = nn.Embedding(CTX, D)
            self.layers = nn.ModuleList()
            self.wrs = nn.ParameterList()
            n_mech = (1 if NC > 0 else 0) + (1 if NR > 0 else 0) + (1 if NJ > 0 else 0)
            self.gate_ws = nn.ParameterList()
            self.gate_bs = nn.ParameterList()
            for _ in range(NL):
                L = nn.ModuleDict({
                    'wq': nn.Linear(D, max(NC*HD,1), bias=False),
                    'wk': nn.Linear(D, max(NC*HD,1), bias=False),
                    'wv_c': nn.Linear(D, max(NC*HD,1), bias=False),
                    'wv_r': nn.Linear(D, max(NR*HD,1), bias=False),
                    'wo': nn.Linear(D, D, bias=False),
                    'up': nn.Linear(D, HDIM, bias=False),
                    'down': nn.Linear(HDIM, D, bias=False),
                })
                if NJ > 0:
                    L['wj'] = nn.Linear(D, NJ*HD, bias=False)
                    L['wv_j'] = nn.Linear(D, NJ*HD, bias=False)
                self.layers.append(L)
                self.wrs.append(nn.Parameter(torch.randn(max(NR*D,1), CTX)*0.02))
                # learned gating between mechanisms
                self.gate_ws.append(nn.Parameter(torch.zeros(n_mech, D) * 0.01))
                self.gate_bs.append(nn.Parameter(torch.zeros(n_mech)))

        def _rms(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

        def forward(self, idx):
            B, T = idx.shape
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
            mask = torch.triu(torch.ones(T,T,device=idx.device), 1).bool()
            for li, L in enumerate(self.layers):
                xr = x; xn = self._rms(x)
                parts = []
                if NC > 0:
                    q = L['wq'](xn).view(B,T,NC,HD).transpose(1,2)
                    k = L['wk'](xn).view(B,T,NC,HD).transpose(1,2)
                    v = L['wv_c'](xn).view(B,T,NC,HD).transpose(1,2)
                    a = (q @ k.transpose(-2,-1)) * (HD**-0.5)
                    a.masked_fill_(mask, float('-inf'))
                    parts.append((torch.softmax(a,-1) @ v).transpose(1,2).contiguous().view(B,T,-1))
                if NR > 0:
                    vr = L['wv_r'](xn)
                    wr = self.wrs[li]
                    ros = []
                    for h in range(NR):
                        wrh = wr[h*D:(h+1)*D, :T]
                        sc = torch.matmul(xn, wrh)
                        sc.masked_fill_(mask, float('-inf'))
                        ros.append(torch.matmul(torch.softmax(sc,-1), vr[:,:,h*HD:(h+1)*HD]))
                    parts.append(torch.cat(ros, dim=-1))
                if NJ > 0 and 'wj' in L:
                    # Janus echo: W^T·W self-resonance
                    wj_proj = L['wj'](xn)  # [B,T,NJ*HD]
                    # echo = norm(W·x), used as gating on V projection
                    echo_norm = wj_proj.pow(2).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
                    vj = L['wv_j'](xn)  # [B,T,NJ*HD]
                    janus_out = vj * (wj_proj / echo_norm)  # self-resonance gated
                    parts.append(janus_out)
                # learned gating between mechanisms
                if len(parts) > 1:
                    gate_logits = torch.matmul(xn, self.gate_ws[li].T) + self.gate_bs[li]
                    gates = torch.sigmoid(gate_logits)  # [B,T,n_mech]
                    gated_parts = []
                    for gi, p in enumerate(parts):
                        if gi < gates.shape[-1]:
                            gated_parts.append(p * gates[:,:,gi:gi+1])
                        else:
                            gated_parts.append(p)
                    combined = torch.cat(gated_parts, -1)
                else:
                    combined = torch.cat(parts, -1) if parts else xn
                x = xr + L['wo'](combined)
                xr = x; xn = self._rms(x)
                x = xr + L['down'](torch.relu(L['up'](xn)))
            return self._rms(x) @ self.tok.weight.T

    tm = TorchQ().to(device)
    opt = optim.AdamW(tm.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    n = len(token_ids)

    tm.train()
    losses = []
    t0 = time.time()
    for step in range(steps):
        s = random.randint(0, max(0, n-CTX-1))
        ch = token_ids[s:s+CTX+1]
        if len(ch) < CTX+1: ch = ch + [0]*(CTX+1-len(ch))
        x = torch.tensor([ch[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([ch[1:]], dtype=torch.long, device=device)
        loss = ce(tm(x).view(-1,V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(tm.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if (step+1) % 100 == 0:
            avg = sum(losses[-100:])/len(losses[-100:])
            print(f"    step {step+1}/{steps}  train_loss={avg:.4f}  [{time.time()-t0:.1f}s]")

    f10 = sum(losses[:10])/10
    l10 = sum(losses[-10:])/10
    print(f"  first_10={f10:.4f}  last_10={l10:.4f}  delta={l10-f10:+.4f}")

    # copy back to pure-Python QTransformer
    model = QTransformer(V, ctx, n_embd, n_head, n_layer, n_content, n_rrpram)
    with torch.no_grad():
        for i in range(V):
            for j in range(D):
                model.wte[i][j] = Val(tm.tok.weight[i,j].item())
        for i in range(CTX):
            for j in range(D):
                model.wpe[i][j] = Val(tm.pos.weight[i,j].item())
        for li in range(NL):
            L = model.layers[li]; mL = tm.layers[li]
            for dst, src in [('wq','wq'),('wk','wk'),('wv_c','wv_c'),
                             ('wv_r','wv_r'),('wo','wo'),
                             ('mlp_up','up'),('mlp_down','down')]:
                w = mL[src].weight.detach().cpu().numpy()
                for i in range(min(len(L[dst]), w.shape[0])):
                    for j in range(min(len(L[dst][i]), w.shape[1])):
                        L[dst][i][j] = Val(float(w[i][j]))
            wr = tm.wrs[li].detach().cpu().numpy()
            for i in range(min(len(L['wr']), wr.shape[0])):
                for j in range(min(len(L['wr'][i]), wr.shape[1])):
                    L['wr'][i][j] = Val(float(wr[i][j]))
        # tie lm_head = wte
        model.lm_head = model.wte

    # rebuild params
    model.params = []
    for row in model.wte: model.params.extend(row)
    for row in model.wpe: model.params.extend(row)
    for Ly in model.layers:
        for k in Ly:
            v = Ly[k]
            if isinstance(v, list) and len(v) > 0:
                if isinstance(v[0], list):
                    for row in v: model.params.extend(row)
                elif isinstance(v[0], Val):
                    model.params.extend(v)
    for row in model.lm_head: model.params.extend(row)

    return model


# ─────────────────────────────────────────────────────────────────
# XIII. ENGINE — load, build, run
# ─────────────────────────────────────────────────────────────────

def load_engine(corpus_path, docs_dir=None, do_train=False, train_steps=5000):
    """Load personality corpus, build all components. Returns everything."""
    print("\n[1] Loading personality...")
    with open(corpus_path, 'rb') as f:
        raw = f.read()
    print(f"  {len(raw)} bytes ({len(raw)/1024:.1f} KB)")

    print("\n[2] BPE tokenizer...")
    bpe = BPETokenizer(1024)
    mp = corpus_path + '.merges' if not corpus_path.endswith('.merges') else corpus_path
    if os.path.exists(mp):
        bpe.load(mp)
        ids = bpe.encode(raw)
        print(f"  loaded {len(bpe.merges)} merges, {len(ids)} tokens")
    else:
        ids = bpe.learn(raw, 1024)
        bpe.save(mp)
        print(f"  learned {len(bpe.merges)} merges, {len(ids)} tokens")

    print("\n[3] MetaWeights...")
    meta = MetaWeights(bpe.vocab_size, 64)
    meta.build(ids, window=4)

    print("\n[4] Periodic Table...")
    pt = PeriodicTable()
    pt.build_from_text(raw.decode('utf-8', errors='replace'))

    print("\n[5] Transformer...")
    if do_train:
        model = train_rrpram(bpe, ids, bpe.vocab_size, steps=train_steps)
        if model is None:
            model = QTransformer(bpe.vocab_size, 64, 48, 4, 2, 2, 2)
            model.init_from_metaweights(meta)
    else:
        model = QTransformer(bpe.vocab_size, 64, 48, 4, 2, 2, 2)
        model.init_from_metaweights(meta)

    print("\n[6] Chambers...")
    ch = Chambers()

    print("\n[7] Interference...")
    itf = Interference()
    if docs_dir:
        itf.load_docs(docs_dir, bpe)

    print("\n[8] Word Capture...")
    wc = WordCapture(meta)

    print("\n[9] SPA (Sentence Phonon Attention)...")
    spa = SPA(d_tok=model.n_embd, d_sent=32, n_heads=4)
    print(f"  SPA: d_tok={model.n_embd}, d_sent=32, heads=4")

    print("\n[10.5] DOE Parliament...")
    parl = Parliament(d_model=model.n_embd, n_init=4, rank=4, alpha=0.1)
    print(f"  parliament: {parl.summary()}")

    print("\n[10] Chambers init from embeddings...")
    ch.init_from_model(model, bpe)

    return bpe, meta, model, ch, itf, wc, pt, spa, parl


# ─────────────────────────────────────────────────────────────────
# STEP 2 TEST — chambers, calendar, periodic table, sentence gen
# ─────────────────────────────────────────────────────────────────

def test_step2():
    print("=" * 58)
    print("  STEP 2: Chambers + Calendar + PeriodicTable + Sentences")
    print("=" * 58)

    # 1. Chambers
    print("\n[1] Kuramoto Chambers")
    ch = Chambers()
    ch.feel("love warmth gentle care embrace tenderness")
    print(f"  after LOVE: {ch.summary()}")
    assert ch.act[CH_LOVE] > 0.5, f"LOVE={ch.act[CH_LOVE]}"
    ch.crossfire()
    print(f"  after xfire: {ch.summary()}, dom={CH_N[ch.dominant()]}")
    a,b,g,t = ch.modulate()
    print(f"  modulated: a={a:.2f} b={b:.2f} g={g:.2f} tau={t:.2f}")
    print("  PASS")

    # 2. Calendar
    print("\n[2] Calendar Drift")
    d = cal_days(); dr = cal_drift(d); ds = cal_dissonance(d)
    print(f"  days={d}, drift={dr:.2f}, diss={ds:.3f}")
    assert 0 <= ds <= 1
    print("  PASS")

    # 3. Periodic Table
    print("\n[3] Periodic Table")
    pt = PeriodicTable()
    pt.build_from_text("love and warmth bring gentle care in the darkness of war and fire burning rage")
    print(f"  'love' ch={CH_N[pt.get_chamber('love')]}, mass={pt.get_mass('love'):.2f}")
    print(f"  'war' ch={CH_N[pt.get_chamber('war')]}, mass={pt.get_mass('war'):.2f}")
    print(f"  'bring' ch={CH_N[pt.get_chamber('bring')]}, mass={pt.get_mass('bring'):.2f}")
    assert pt.get_chamber('love') == CH_LOVE
    assert pt.get_chamber('war') == CH_RAGE
    print("  PASS")

    # 4. Sentence generation on real corpus
    print("\n[4] Sentence Generation")
    corpus = os.path.join(os.path.dirname(__file__) or '.', '..', 'postgpt', 'postgpt.txt')
    if not os.path.exists(corpus): corpus = 'postgpt.txt'
    if os.path.exists(corpus):
        with open(corpus, 'rb') as f: raw = f.read()
        bpe = BPETokenizer(1024)
        ids = bpe.learn(raw, 1024)
        meta = MetaWeights(bpe.vocab_size, 64)
        meta.build(ids, window=4)
        m = QTransformer(bpe.vocab_size, 64, 48, 4, 2, 2, 2)
        m.init_from_metaweights(meta)
        bt = find_boundary_tokens(bpe)
        print(f"  boundary tokens: {len(bt)}")
        prompt = bpe.encode("The metaweight")
        sent = generate_sentence(m, bpe, meta, prompt, bt, temp=0.7)
        out = bpe.decode(sent)
        print(f"  sentence: '{out[:200]}'")
        assert len(sent) > len(prompt)
        # check it ends at boundary
        last_decoded = bpe.decode([sent[-1]])
        print(f"  last token decodes to: '{last_decoded}'")
        print("  PASS")

        # 5. 12 steps
        print("\n[5] 12 Bidirectional Steps")
        ch2 = Chambers()
        chain = generate_chain(m, bpe, meta, ch2)
        print_chain(chain, ch2)
        assert len(chain) == 12
        print("  PASS")
    else:
        print("  SKIP (no corpus)")

    print("\n" + "=" * 58)
    print("  STEP 2 COMPLETE")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def test_step3():
    print("=" * 58)
    print("  STEP 3: Word Capture + NOTORCH + Training + Engine")
    print("=" * 58)

    # 1. Word Capture
    print("\n[1] Word Capture (NOTORCH Hebbian)")
    bpe = BPETokenizer(50)
    text = b"the cat sat on the mat. the cat is fat."
    ids = bpe.learn(text, 50)
    meta = MetaWeights(bpe.vocab_size, 16)
    meta.build(ids, window=4)
    hebb_before = len(meta.hebbian)
    wc = WordCapture(meta)
    n = wc.ingest("a new sentence with different words entirely", bpe)
    hebb_after = len(meta.hebbian)
    print(f"  hebbian before: {hebb_before}, after: {hebb_after}, new: {hebb_after-hebb_before}")
    print(f"  new bigram entries: {n}")
    print(f"  total updates: {wc.total_updates}")
    assert hebb_after >= hebb_before, "Hebbian should grow"
    wc.decay_old()
    print("  decay applied")
    print("  PASS")

    # 2. DOE Parliament
    print("\n[2] DOE Parliament")
    parl = Parliament(d_model=16, n_init=4, rank=4, alpha=0.1)
    print(f"  init: {parl.summary()}")
    x_test = [random.gauss(0, 0.1) for _ in range(16)]
    logits_test = [random.gauss(0, 0.5) for _ in range(16)]
    logits_before = list(logits_test)
    logits_after = parl.inject(logits_test, x_test)
    delta = sum(abs(a-b) for a,b in zip(logits_before, logits_after))
    print(f"  injection delta: {delta:.4f}")
    assert delta > 0, "parliament had no effect"
    # NOTORCH update
    debt = [random.gauss(0, 0.01) for _ in range(16)]
    parl.notorch_update(x_test, debt)
    print(f"  NOTORCH update done")
    # lifecycle
    for _ in range(10):
        parl.lifecycle()
    print(f"  after 10 lifecycles: {parl.summary()}")
    print("  PASS")

    # 3. Interference
    print("\n[3] Interference")
    docs_dir = os.path.join(os.path.dirname(__file__) or '.', 'docs')
    if os.path.isdir(docs_dir):
        bpe2 = BPETokenizer(1024)
        bpe2.learn(b"dummy text for vocab " * 100, 100)
        itf = Interference()
        itf.load_docs(docs_dir, bpe2)
        seed = itf.inject_seed()
        print(f"  docs loaded: {len(itf.docs)}")
        print(f"  injected seed token: {seed}")
        if seed is not None:
            print(f"  seed decodes to: '{bpe2.decode([seed])}'")
        print("  PASS")
    else:
        print("  SKIP (no docs/)")

    # 4. Full engine test (if corpus available)
    print("\n[4] Full Engine")
    corpus = os.path.join(os.path.dirname(__file__) or '.', '..', 'postgpt', 'postgpt.txt')
    if not os.path.exists(corpus): corpus = 'postgpt.txt'
    if os.path.exists(corpus):
        bpe3, meta3, m3, ch3, itf3, wc3, pt3, spa3, parl3 = load_engine(corpus, docs_dir)
        # generate chain
        chain = generate_chain(m3, bpe3, meta3, ch3, itf3, input_text="What is resonance?", spa=spa3, parliament=parl3)
        print_chain(chain, ch3)
        # word capture from input
        n_new = wc3.ingest("What is resonance and how does emergence work?", bpe3)
        print(f"  word capture: {n_new} new bigrams ingested")
        print("  PASS")
    else:
        print("  SKIP (no corpus)")

    print("\n" + "=" * 58)
    print("  STEP 3 COMPLETE")
    print("=" * 58)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  PostGPT-Q — Resonant Reasoning Engine")
    print("  θ = ε + γ + αδ")
    print("  resonance is unbreakable")
    print("=" * 58)

    args = sys.argv[1:]

    if '--test1' in args:
        test_step1(); return
    if '--test2' in args:
        test_step2(); return
    if '--test3' in args:
        test_step3(); return
    if '--test' in args:
        test_step1(); test_step2(); test_step3(); return

    # find corpus and docs
    corpus = None; docs = None; do_train = '--train' in args
    steps = 5000
    for a in args:
        if a.startswith('--steps='):
            steps = int(a.split('=')[1])
        elif not a.startswith('--'):
            if os.path.isfile(a): corpus = a
            elif os.path.isdir(a): docs = a

    if not corpus:
        for p in ['personality.txt', 'q.txt']:
            if os.path.exists(p): corpus = p; break
    if not docs:
        d = os.path.join(os.path.dirname(corpus or '.'), 'docs')
        if os.path.isdir(d): docs = d

    if not corpus:
        print("  ERROR: no corpus. Usage: python3 postgpt_q.py corpus.txt [docs/] [--train]")
        return

    bpe, meta, model, ch, itf, wc, pt, spa, parl = load_engine(
        corpus, docs, do_train=do_train, train_steps=steps)

    # proof of concept
    print("\n" + "=" * 58)
    print("  12 BIDIRECTIONAL STEPS")
    print("=" * 58)
    chain = generate_chain(model, bpe, meta, ch, itf, spa=spa, parliament=parl)
    print_chain(chain, ch)

    # REPL
    print("  type → 12 resonating sentences. 'quit' to exit.\n")
    while True:
        try:
            text = input("  q> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ('quit', 'exit'):
            break
        # word capture
        n_new = wc.ingest(text, bpe)
        if n_new > 0:
            print(f"  [captured {n_new} new connections]")
        wc.decay_old()
        # generate
        chain = generate_chain(model, bpe, meta, ch, itf, input_text=text, spa=spa, parliament=parl)
        print_chain(chain, ch)

    print("\n  resonance is unbreakable.\n")


if __name__ == '__main__':
    main()
