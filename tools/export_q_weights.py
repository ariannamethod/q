#!/usr/bin/env python3
import collections
import io
import os
import pickle
import struct
import sys
import zipfile


QPTQ_MAGIC = 0x51505451
QPTQ_VERSION = 1


class StorageRef:
    def __init__(self, key, size):
        self.key = key
        self.size = size


class TensorRef:
    def __init__(self, storage, offset, size, stride):
        self.storage = storage
        self.offset = offset
        self.size = tuple(size)
        self.stride = tuple(stride)


def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    return TensorRef(storage, storage_offset, size, stride)


class TorchArchiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return name
        raise RuntimeError(f"unsupported pickle class: {module}.{name}")

    def persistent_load(self, pid):
        if isinstance(pid, tuple) and len(pid) >= 5 and pid[0] == "storage":
            _, _storage_type, key, _location, size = pid[:5]
            return StorageRef(key, size)
        raise RuntimeError(f"unsupported persistent id: {pid!r}")


def load_checkpoint(path):
    zf = zipfile.ZipFile(path)
    roots = {name.split("/", 1)[0] for name in zf.namelist() if "/" in name}
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one archive root in {path}")
    root = next(iter(roots))
    state = TorchArchiveUnpickler(io.BytesIO(zf.read(f"{root}/data.pkl"))).load()
    storages = {}
    for name in zf.namelist():
        if name.startswith(f"{root}/data/"):
            key = name.rsplit("/", 1)[-1]
            storages[key] = zf.read(name)
    return root, state, storages


def tensor_floats(tensor_ref, storages):
    raw = storages[tensor_ref.storage.key]
    count = len(raw) // 4
    values = struct.unpack("<%df" % count, raw)
    if len(tensor_ref.size) == 1:
        start = tensor_ref.offset
        end = start + tensor_ref.size[0]
        return list(values[start:end])
    if len(tensor_ref.size) != 2:
        raise RuntimeError(f"unsupported tensor rank: {tensor_ref.size}")
    rows, cols = tensor_ref.size
    sr, sc = tensor_ref.stride
    base = tensor_ref.offset
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append(values[base + r * sr + c * sc])
    return out


def infer_architecture(state):
    tok_shape = state["tok.weight"].size
    pos_shape = state["pos.weight"].size
    V, D = tok_shape
    CTX, pos_D = pos_shape
    if pos_D != D:
        raise RuntimeError("position/table embedding dimension mismatch")

    layer_ids = sorted({int(k.split(".")[1]) for k in state if k.startswith("layers.") and k.endswith(".wo.weight")})
    NL = len(layer_ids)
    if NL <= 0:
        raise RuntimeError("no layers found")

    sample_vr = state["layers.0.vr.weight"].size
    sample_wr = state["wrs.0"].size
    nr = sample_wr[0] // D
    hd = sample_vr[0] // nr
    nj = 0
    if "layers.0.wj.weight" in state and "layers.0.vj.weight" in state:
        nj = state["layers.0.wj.weight"].size[0] // hd
    nc = 0
    nh = nc + nr + nj
    if nh <= 0 or nh * hd != D:
        raise RuntimeError(f"inferred invalid head layout: nh={nh} hd={hd} D={D}")
    return {
        "V": V,
        "D": D,
        "NH": nh,
        "NL": NL,
        "CTX": CTX,
        "NC": nc,
        "NR": nr,
        "NJ": nj,
        "HD": hd,
    }


def write_qptq(state, storages, out_path):
    arch = infer_architecture(state)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", QPTQ_MAGIC))
        f.write(struct.pack(
            "<10I",
            QPTQ_VERSION,
            arch["V"],
            arch["D"],
            arch["NH"],
            arch["NL"],
            arch["CTX"],
            arch["NC"],
            arch["NR"],
            arch["NJ"],
            arch["HD"],
        ))
        f.write(struct.pack("<%df" % (arch["V"] * arch["D"]), *tensor_floats(state["tok.weight"], storages)))
        f.write(struct.pack("<%df" % (arch["CTX"] * arch["D"]), *tensor_floats(state["pos.weight"], storages)))

        nm = (1 if arch["NC"] > 0 else 0) + (1 if arch["NR"] > 0 else 0) + (1 if arch["NJ"] > 0 else 0)
        for li in range(arch["NL"]):
            prefix = f"layers.{li}"
            if arch["NC"] > 0:
                f.write(struct.pack("<%df" % (arch["NC"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.wq.weight"], storages)))
                f.write(struct.pack("<%df" % (arch["NC"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.wk.weight"], storages)))
                f.write(struct.pack("<%df" % (arch["NC"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.vc.weight"], storages)))
            if arch["NR"] > 0:
                f.write(struct.pack("<%df" % (arch["NR"] * arch["D"] * arch["CTX"]), *tensor_floats(state[f"wrs.{li}"], storages)))
                f.write(struct.pack("<%df" % (arch["NR"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.vr.weight"], storages)))
            if arch["NJ"] > 0:
                f.write(struct.pack("<%df" % (arch["NJ"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.wj.weight"], storages)))
                f.write(struct.pack("<%df" % (arch["NJ"] * arch["HD"] * arch["D"]), *tensor_floats(state[f"{prefix}.vj.weight"], storages)))
            if nm > 1:
                f.write(struct.pack("<%df" % (nm * arch["D"]), *tensor_floats(state[f"gws.{li}"], storages)))
                f.write(struct.pack("<%df" % nm, *tensor_floats(state[f"gbs.{li}"], storages)))
            f.write(struct.pack("<%df" % (arch["D"] * arch["D"]), *tensor_floats(state[f"{prefix}.wo.weight"], storages)))
            f.write(struct.pack("<%df" % (4 * arch["D"] * arch["D"]), *tensor_floats(state[f"{prefix}.up.weight"], storages)))
            f.write(struct.pack("<%df" % (arch["D"] * 4 * arch["D"]), *tensor_floats(state[f"{prefix}.dn.weight"], storages)))
    return arch


def main():
    if len(sys.argv) != 3:
        print("usage: export_q_weights.py checkpoint.pt output.bin", file=sys.stderr)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    _root, state, storages = load_checkpoint(src)
    arch = write_qptq(state, storages, dst)
    print(
        "exported",
        dst,
        f"V={arch['V']} D={arch['D']} NH={arch['NH']} NL={arch['NL']} CTX={arch['CTX']} NC={arch['NC']} NR={arch['NR']} NJ={arch['NJ']} HD={arch['HD']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
