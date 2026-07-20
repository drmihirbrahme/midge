"""midgepack — the midge container format, in Python.

Single source of truth on the tooling side for:
  * quantization codecs (q8r, q4g32, mxfp4) — encode from f32, decode to f32
  * dense.midge / experts.midge writers and readers
  * the deterministic expert layout (MUST mirror engine/mten.h wt_expert_layout)
  * a minimal, torch-free safetensors reader (handles BF16/F32/F16/U8/I64)

No dependency beyond numpy.
"""
from __future__ import annotations
import json
import mmap
import os
import struct
import numpy as np
import os as _os


def midge_home():
    """Writable base dir for built engine + models (packaged app friendly)."""
    h = _os.environ.get("MIDGE_HOME")
    if h:
        _os.makedirs(h, exist_ok=True)
        return h
    return _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


def engine_path():
    """Path to the midged binary: explicit env, else repo/home root."""
    e = _os.environ.get("MIDGE_ENGINE")
    if e:
        return e
    exe = "midged.exe" if _os.name == "nt" else "midged"
    for base in (midge_home(),
                 _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))):
        p = _os.path.join(base, exe)
        if _os.path.exists(p):
            return p
    return _os.path.join(midge_home(), exe)


def models_dir():
    d = _os.environ.get("MIDGE_MODELS_DIR") or _os.path.join(midge_home(), "models")
    return d

ALIGN = 64
FP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def align64(x: int) -> int:
    return (x + ALIGN - 1) & ~(ALIGN - 1)


# --------------------------------------------------------------- codecs
def q8r_encode(w: np.ndarray):
    """int8 per-row symmetric. Returns (data u8 view, scales f32)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scale[:, None]), -127, 127).astype(np.int8)
    return q.tobytes(), scale.tobytes()


def q8r_decode(data: bytes, scales: bytes, rows: int, cols: int) -> np.ndarray:
    q = np.frombuffer(data, dtype=np.int8).reshape(rows, cols).astype(np.float32)
    s = np.frombuffer(scales, dtype=np.float32).reshape(rows, 1)
    return q * s


def _pack_nibbles(codes: np.ndarray) -> bytes:
    """codes: (rows, cols) uint8 in 0..15; low nibble = even column."""
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    return ((hi << 4) | lo).astype(np.uint8).tobytes()


def _unpack_nibbles(data: bytes, rows: int, cols: int) -> np.ndarray:
    b = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols // 2)
    out = np.empty((rows, cols), dtype=np.uint8)
    out[:, 0::2] = b & 15
    out[:, 1::2] = b >> 4
    return out


def q4g32_encode(w: np.ndarray):
    """uniform symmetric int4 (stored biased +8), group=32, f16 scales."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, cols = w.shape
    assert cols % 32 == 0, "q4g32 needs cols % 32 == 0"
    g = w.reshape(rows, cols // 32, 32)
    amax = np.abs(g).max(axis=2)
    scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(g / scale[:, :, None]), -7, 7).astype(np.int8) + 8
    codes = q.reshape(rows, cols).astype(np.uint8)
    return _pack_nibbles(codes), scale.astype(np.float16).tobytes()


def q4g32_decode(data: bytes, scales: bytes, rows: int, cols: int) -> np.ndarray:
    codes = _unpack_nibbles(data, rows, cols).astype(np.int32) - 8
    s = np.frombuffer(scales, dtype=np.float16).astype(np.float32)
    s = s.reshape(rows, cols // 32, 1)
    return (codes.reshape(rows, cols // 32, 32) * s).reshape(rows, cols).astype(np.float32)


def mxfp4_encode(w: np.ndarray):
    """Encode f32 to MXFP4-style: FP4 E2M1 codes + per-32-group f16 scale.
    Used for tests and for requantizing bf16 checkpoints; real MXFP4
    checkpoints are transcoded losslessly by the converter instead."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, cols = w.shape
    assert cols % 32 == 0
    data_parts, scale_parts = [], []
    step = max(1, (1 << 24) // max(cols, 1))  # ~16M elements per chunk
    for r0 in range(0, rows, step):
        g = w[r0:r0 + step].reshape(-1, cols // 32, 32)
        amax = np.abs(g).max(axis=2)
        # true MXFP4: power-of-two (e8m0) shared scale, chosen so |w|/s <= 6
        with np.errstate(divide="ignore"):
            e = np.ceil(np.log2(amax / 6.0))
        e = np.where(amax > 0, e, 0.0)
        scale = np.exp2(np.clip(e, -24, 15)).astype(np.float32)
        norm = g / scale[:, :, None]
        # nearest FP4 value per element
        dist = np.abs(norm[..., None] - FP4_LUT[None, None, None, :])
        codes = dist.argmin(axis=-1).astype(np.uint8)
        data_parts.append(_pack_nibbles(codes.reshape(-1, cols)))
        scale_parts.append(scale.astype(np.float16).tobytes())
    return b"".join(data_parts), b"".join(scale_parts)


def mxfp4_decode(data: bytes, scales: bytes, rows: int, cols: int) -> np.ndarray:
    codes = _unpack_nibbles(data, rows, cols)
    vals = FP4_LUT[codes]
    s = np.frombuffer(scales, dtype=np.float16).astype(np.float32)
    s = s.reshape(rows, cols // 32, 1)
    return (vals.reshape(rows, cols // 32, 32) * s).reshape(rows, cols)


def encode(dt: str, w: np.ndarray):
    if dt == "f32":
        return np.ascontiguousarray(w, dtype=np.float32).tobytes(), b""
    if dt == "q8r":
        return q8r_encode(w)
    if dt == "q4g32":
        return q4g32_encode(w)
    if dt == "mxfp4":
        return mxfp4_encode(w)
    raise ValueError(f"unknown dtype {dt}")


def decode(dt: str, data: bytes, scales: bytes, rows: int, cols: int) -> np.ndarray:
    if dt == "f32":
        return np.frombuffer(data, dtype=np.float32).reshape(rows, cols).copy()
    if dt == "q8r":
        return q8r_decode(data, scales, rows, cols)
    if dt == "q4g32":
        return q4g32_decode(data, scales, rows, cols)
    if dt == "mxfp4":
        return mxfp4_decode(data, scales, rows, cols)
    raise ValueError(f"unknown dtype {dt}")


def blob_sizes(dt: str, rows: int, cols: int):
    """(data_bytes, scale_bytes) — must mirror wt_sizes() in engine/mten.h."""
    if dt == "f32":
        return rows * cols * 4, 0
    if dt == "q8r":
        return rows * cols, rows * 4
    if dt in ("q4g32", "mxfp4"):
        return rows * cols // 2, rows * (cols // 32) * 2
    raise ValueError(dt)


# ------------------------------------------------------- expert layout
class ExpertLayout:
    """Deterministic offsets — MUST mirror wt_expert_layout() in mten.h."""

    MATS = ("gate", "up", "down")
    PARTS = ("data", "scales", "bias")

    def __init__(self, dt: str, hidden: int, ffn: int, n_experts: int, n_layers: int):
        self.dt, self.hidden, self.ffn = dt, hidden, ffn
        self.n_experts, self.n_layers = n_experts, n_layers
        rows = {"gate": ffn, "up": ffn, "down": hidden}
        cols = {"gate": hidden, "up": hidden, "down": ffn}
        self.rows, self.cols = rows, cols
        off = 0
        self.off = {}
        for m in self.MATS:
            db, sb = blob_sizes(dt, rows[m], cols[m])
            self.off[(m, "data")] = off
            off = align64(off + db)
            self.off[(m, "scales")] = off
            off = align64(off + sb)
            self.off[(m, "bias")] = off
            off = align64(off + rows[m] * 4)
        self.expert_stride = align64(off)
        self.layer_stride = self.expert_stride * n_experts
        self.total = self.layer_stride * n_layers

    def offset(self, layer: int, expert: int, mat: str, part: str) -> int:
        return layer * self.layer_stride + expert * self.expert_stride + self.off[(mat, part)]


# ------------------------------------------------------------ writers
def _write_header(f, hdr: dict):
    blob = json.dumps(hdr).encode()
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    pad = align64(8 + len(blob)) - (8 + len(blob))
    f.write(b"\0" * pad)
    return align64(8 + len(blob))


class DenseWriter:
    """Accumulates tensors in memory-order and writes dense.midge.
    Dense sets are small (a couple of GB at most), but we still stream:
    add() writes immediately at growing offsets; header is written last
    to a reserved slot? No — we buffer index and write data to a temp file,
    then concatenate. Simpler and safe for <3 GB."""

    def __init__(self, path: str, spec: dict):
        self.path = path
        self.spec = spec
        self.index = {}
        self.tmp = path + ".data.tmp"
        self.f = open(self.tmp, "wb")
        self.off = 0

    def _put(self, blob: bytes) -> int:
        at = self.off
        self.f.write(blob)
        pad = align64(len(blob)) - len(blob)
        if pad:
            self.f.write(b"\0" * pad)
        self.off = align64(self.off + len(blob))
        return at

    def add(self, name: str, w: np.ndarray, dt: str = "f32"):
        if w.ndim == 1:
            assert dt == "f32", "vectors are stored f32"
            data, scales = np.ascontiguousarray(w, dtype=np.float32).tobytes(), b""
            shape = [int(w.shape[0])]
        else:
            data, scales = encode(dt, w)
            shape = [int(w.shape[0]), int(w.shape[1])]
        e = {"dt": dt, "shape": shape, "off": self._put(data)}
        if scales:
            e["soff"] = self._put(scales)
        self.index[name] = e

    def close(self):
        self.f.close()
        hdr = {"midge": 1, "kind": "dense", "spec": self.spec, "tensors": self.index}
        with open(self.path, "wb") as out:
            _write_header(out, hdr)
            with open(self.tmp, "rb") as t:
                while True:
                    chunk = t.read(1 << 22)
                    if not chunk:
                        break
                    out.write(chunk)
        os.remove(self.tmp)


class ExpertWriter:
    """Sparse random-offset writer for experts.midge (resumable)."""

    def __init__(self, path: str, layout: ExpertLayout):
        self.layout = layout
        hdr = {"midge": 1, "kind": "experts", "dt": layout.dt}
        exists = os.path.exists(path)
        self.f = open(path, "r+b" if exists else "w+b")
        self.base = _write_header(self.f, hdr)  # header is deterministic; rewrite is idempotent
        self.f.truncate(self.base + layout.total)

    def put(self, layer: int, expert: int, mat: str, blob_data: bytes,
            blob_scales: bytes, bias_f32: bytes):
        L = self.layout
        db, sb = blob_sizes(L.dt, L.rows[mat], L.cols[mat])
        assert len(blob_data) == db, (mat, len(blob_data), db)
        assert len(blob_scales) == sb
        assert len(bias_f32) == L.rows[mat] * 4
        self.f.seek(self.base + L.offset(layer, expert, mat, "data"))
        self.f.write(blob_data)
        if sb:
            self.f.seek(self.base + L.offset(layer, expert, mat, "scales"))
            self.f.write(blob_scales)
        self.f.seek(self.base + L.offset(layer, expert, mat, "bias"))
        self.f.write(bias_f32)

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


# ------------------------------------------------------------- readers
def _read_header(f):
    (hl,) = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(hl))
    return hdr, align64(8 + hl)


def read_dense(path: str):
    """Returns (spec, dict name -> f32 ndarray). Dequantizes everything."""
    with open(path, "rb") as f:
        hdr, base = _read_header(f)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    out = {}
    for name, e in hdr["tensors"].items():
        shape = e["shape"]
        if len(shape) == 1:
            out[name] = np.frombuffer(mm, dtype=np.float32, count=shape[0],
                                      offset=base + e["off"]).copy()
        else:
            rows, cols = shape
            db, sb = blob_sizes(e["dt"], rows, cols)
            data = mm[base + e["off"]: base + e["off"] + db]
            scales = mm[base + e["soff"]: base + e["soff"] + sb] if sb else b""
            out[name] = decode(e["dt"], data, scales, rows, cols)
    return hdr["spec"], out


def read_experts(path: str, spec: dict):
    """Returns (dt, getter(layer, expert) -> dict with f32 gate/up/down (+biases))."""
    with open(path, "rb") as f:
        hdr, base = _read_header(f)
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    dt = hdr["dt"]
    L = ExpertLayout(dt, spec["hidden"], spec["moe"]["ffn"],
                     spec["moe"]["experts"], spec["n_layers"])

    def get(layer: int, expert: int):
        out = {}
        for mat in L.MATS:
            rows, cols = L.rows[mat], L.cols[mat]
            db, sb = blob_sizes(dt, rows, cols)
            o = base + L.offset(layer, expert, mat, "data")
            so = base + L.offset(layer, expert, mat, "scales")
            bo = base + L.offset(layer, expert, mat, "bias")
            out[mat] = decode(dt, mm[o:o + db], mm[so:so + sb] if sb else b"", rows, cols)
            out[mat + "_b"] = np.frombuffer(mm, dtype=np.float32, count=rows, offset=bo).copy()
        return out

    return dt, get


# ------------------------------------------- minimal safetensors reader
_ST_DTYPES = {
    "F32": (np.float32, 4), "F16": (np.float16, 2), "BF16": (None, 2),
    "U8": (np.uint8, 1), "I8": (np.int8, 1), "I32": (np.int32, 4),
    "I64": (np.int64, 8), "U16": (np.uint16, 2), "F64": (np.float64, 8),
}


class SafeTensors:
    """Torch-free reader. Values are returned as numpy arrays; BF16 is
    upcast to f32. Uses memmap so reading one tensor touches only its bytes."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            (hl,) = struct.unpack("<Q", f.read(8))
            self.meta = json.loads(f.read(hl))
        self.meta.pop("__metadata__", None)
        self.base = 8 + hl
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def names(self):
        return list(self.meta.keys())

    def info(self, name):
        e = self.meta[name]
        return e["dtype"], e["shape"]

    def get(self, name) -> np.ndarray:
        e = self.meta[name]
        dt, shape = e["dtype"], e["shape"]
        start, end = e["data_offsets"]
        raw = self.mm[self.base + start: self.base + end]
        if dt == "BF16":
            u16 = raw.view(np.uint16).astype(np.uint32) << 16
            arr = u16.view(np.float32)
        else:
            np_dt, _ = _ST_DTYPES[dt]
            arr = raw.view(np_dt)
        return arr.reshape(shape)


def write_safetensors(path: str, tensors: dict):
    """tensors: name -> (np array, dtype string). For tests/fixtures."""
    meta, blobs, off = {}, [], 0
    for name, (arr, dt) in tensors.items():
        if dt == "BF16":
            f32 = np.ascontiguousarray(arr, dtype=np.float32)
            u = f32.view(np.uint32)
            # round-to-nearest-even bf16
            rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            raw = rounded.tobytes()
        else:
            np_dt, _ = _ST_DTYPES[dt]
            raw = np.ascontiguousarray(arr, dtype=np_dt).tobytes()
        meta[name] = {"dtype": dt, "shape": list(arr.shape),
                      "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)
