#!/usr/bin/env python3
# See LICENSE for license details.
"""MXINT8 external cross-check against microsoft/microxcaling (PLAN_TEST.md 1.3).

Diffs the in-repo packer + golden GEMM (run on the host via
``tools/mxint8_host_ref.c``, so the *actual project code* is exercised) against
the OCP MX spec authors' reference library, under the frozen policy
(``mxint8_policy.md``: block_size=32, E8M0 scale, int8 payloads with implicit
2^-6, round ties-to-even):

  check 1 (scale rule)  packer scale byte == 127 + floor(log2(block amax)),
                        the spec-6.3 recommended scale, computed exactly from
                        the fp32 inputs (all-zero blocks use the documented
                        neutral-scale convention e = 0).
  check 2 (elements)    packer dequantized values == microxcaling dequantized
                        values, bit-exact in binary64.
  check 3 (GEMM)        golden C output == an independent exact-integer GEMM
                        (Python bignums) over payloads recovered from
                        *microxcaling's* dequantized tensors, replicating the
                        frozen-policy semantics (per-block raw dot, power-of-two
                        scale with sign-magnitude round-nearest-even, int64
                        block clamp, single int32 saturation at readout).

Example:
  python3 tools/mxint8_external_diff.py --backend microxcaling --seed 1 \\
      --m 16 --n 16 --k 64 --round even --json-out build/diff_seed1.json
  python3 tools/mxint8_external_diff.py --sweep --json-out build/diff_sweep.json

Requires a venv with torch (CPU) + microxcaling, e.g. chipyard's ``.mx-venv``.
Note: microxcaling pins torch==2.2.0, which has no wheel for this venv's
Python; it is installed with --no-deps and runs on a newer CPU torch. The
exact versions are recorded in every JSON report.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(TOOLS_DIR)
BLOCK = 32
FRAC_BITS = 6
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
INT32_MIN, INT32_MAX = -(2**31), 2**31 - 1

MISMATCH_DETAIL_LIMIT = 5


def build_host_ref():
    """Compile tools/mxint8_host_ref.c against the staged MX DIM=32 headers.

    Returns the path to the binary, rebuilding it when any source is newer.
    The MX params header is staged as include/gemmini_params.h in a private
    directory, exactly as the baremetal test build stages it.
    """
    out = os.path.join(TESTS_DIR, "build", "mxint8_host_ref")
    srcs = [
        os.path.join(TOOLS_DIR, "mxint8_host_ref.c"),
        os.path.join(TESTS_DIR, "include", "gemmini_params_mxint8_dim32.h"),
        os.path.join(TESTS_DIR, "include", "mxint8_golden.h"),
        os.path.join(TESTS_DIR, "include", "mxint8_pack.h"),
    ]
    if os.path.exists(out) and all(os.path.getmtime(out) >= os.path.getmtime(s) for s in srcs):
        return out

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mxint8_stage_") as stage:
        inc = os.path.join(stage, "include")
        os.makedirs(inc)
        shutil.copy(srcs[1], os.path.join(inc, "gemmini_params.h"))
        shutil.copy(srcs[2], inc)
        shutil.copy(srcs[3], inc)
        subprocess.run(["gcc", "-O2", "-I", stage, srcs[0], "-o", out], check=True)
    return out


def gen_inputs(rng, m, n, k, mode):
    """Seeded fp32 A[m,k], B[k,n] with per-MX-block exponent spread.

    mode 'random': uniform(-1,1) per element, one random power-of-two factor
    2^j, j in [-16,16], per MX block.
    mode 'band': additionally forces each block's amax into the old packer's
    divergence band (1.984375*2^j, 2*2^j), where the pre-P1 scale rule
    provably picked different bits than the spec-6.3 rule.
    """
    kb = (k + BLOCK - 1) // BLOCK

    def fill_rows(rows):
        out = np.zeros((rows, kb * BLOCK), dtype=np.float64)
        for r in range(rows):
            for b in range(kb):
                j = rng.integers(-16, 17)
                vals = rng.uniform(-1.0, 1.0, BLOCK) * (2.0**j)
                if mode == "band":
                    vals *= 0.9  # keep the forced element the unique amax
                    frac = rng.uniform(0.001, 0.999)
                    mag = (1.984375 + frac * (2.0 - 1.984375)) * (2.0**j)
                    sign = 1.0 if rng.integers(0, 2) else -1.0
                    vals[rng.integers(0, BLOCK)] = sign * mag
                out[r, b * BLOCK:(b + 1) * BLOCK] = vals
        return out[:, :k].astype(np.float32)

    a = fill_rows(m)
    b = fill_rows(n).T.copy()  # B blocks run down K within a column
    return a, b


def expected_scale_byte(block_vals):
    """Spec-6.3 scale byte for a block of fp32 values (project conventions)."""
    amax = float(np.max(np.abs(block_vals.astype(np.float64))))
    if amax == 0.0:
        return 127  # documented all-zero-block neutral scale e = 0
    _, e = math.frexp(amax)  # amax = f * 2^e, f in [0.5, 1) => floor(log2) = e-1
    return 127 + max(-127, min(127, e - 1))


def run_host_ref(binary, a, b, tmpdir):
    """Run the in-repo packer + golden on A, B; return the five outputs."""
    m, k = a.shape
    n = b.shape[1]
    kb = (k + BLOCK - 1) // BLOCK
    pre = os.path.join(tmpdir, "t")
    a.tofile(pre + ".a.bin")
    b.tofile(pre + ".b.bin")
    subprocess.run([binary, str(m), str(n), str(k), pre + ".a.bin", pre + ".b.bin", pre],
                   check=True)
    return {
        "a_payload": np.fromfile(pre + ".a_payload.bin", dtype=np.int8).reshape(m, k),
        "b_payload": np.fromfile(pre + ".b_payload.bin", dtype=np.int8).reshape(k, n),
        "a_scale": np.fromfile(pre + ".a_scale.bin", dtype=np.uint8).reshape(m, kb),
        "b_scale": np.fromfile(pre + ".b_scale.bin", dtype=np.uint8).reshape(kb, n),
        "c": np.fromfile(pre + ".c.bin", dtype=np.int32).reshape(m, n),
    }


def microxcaling_dequant(rows_mat, round_mode):
    """MXINT8-quantize with microxcaling along axis 1 (blocks of 32), CPU.

    Input shape (rows, k); k is zero-padded to a block multiple (padding
    cannot change a block amax). Returns the dequantized array, trimmed back
    to the input shape, as float64.
    """
    import torch
    from mx.mx_ops import _quantize_mx

    rows, k = rows_mat.shape
    kb = (k + BLOCK - 1) // BLOCK
    padded = np.zeros((rows, kb * BLOCK), dtype=np.float32)
    padded[:, :k] = rows_mat
    q = _quantize_mx(torch.from_numpy(padded), scale_bits=8, elem_format="int8",
                     shared_exp_method="max", axes=[-1], block_size=BLOCK,
                     round=round_mode, custom_cuda=False)
    return q.numpy()[:, :k].astype(np.float64)


def dequant_rows(payload_rows, scale_rows, k):
    """Dequantize int8 payloads (blocks along axis 1) with E8M0 scale rows."""
    out = np.zeros(payload_rows.shape, dtype=np.float64)
    kb = (k + BLOCK - 1) // BLOCK
    for blk in range(kb):
        lo, hi = blk * BLOCK, min((blk + 1) * BLOCK, k)
        e = scale_rows[:, blk].astype(np.int64) - 127
        out[:, lo:hi] = payload_rows[:, lo:hi].astype(np.float64) \
            * np.exp2((e - FRAC_BITS).astype(np.float64))[:, None]
    return out


def rne_right_shift(value, shift):
    """Golden's sign-magnitude round-to-nearest-even right shift (bignum)."""
    if shift <= 0:
        return value << (-shift)
    mag = -value if value < 0 else value
    q, r = mag >> shift, mag & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if r > half or (r == half and (q & 1)):
        q += 1
    return -q if value < 0 else q


def policy_gemm_exact(a_payload, b_payload, a_scale, b_scale, k):
    """Frozen-policy GEMM in exact Python integers (mirrors mxint8_golden.h).

    Per logical block: raw int dot, scale by 2^(eA+eB-12) with sign-magnitude
    RNE for negative shifts and an int64 clamp for positive ones; blocks sum
    into a wide accumulator saturated once to int32 at readout.
    """
    m, n = a_payload.shape[0], b_payload.shape[1]
    kb = (k + BLOCK - 1) // BLOCK
    c = [[0] * n for _ in range(m)]
    ap = a_payload.astype(np.int64)
    bp = b_payload.astype(np.int64)
    for blk in range(kb):
        lo, hi = blk * BLOCK, min((blk + 1) * BLOCK, k)
        raw = ap[:, lo:hi] @ bp[lo:hi, :]  # exact: |raw| <= 32*128^2 = 2^19
        for i in range(m):
            ea = int(a_scale[i, blk]) - 127
            for j in range(n):
                shift = ea + (int(b_scale[blk, j]) - 127) - 2 * FRAC_BITS
                scaled = rne_right_shift(int(raw[i, j]), -shift)
                c[i][j] += max(INT64_MIN, min(INT64_MAX, scaled))
    out = np.array(c, dtype=object)
    return np.clip(out, INT32_MIN, INT32_MAX).astype(np.int32)


def run_case(binary, seed, m, n, k, mode, round_mode):
    """One seeded cross-check; returns a JSON-able result record."""
    rng = np.random.default_rng(seed)
    a, b = gen_inputs(rng, m, n, k, mode)
    kb = (k + BLOCK - 1) // BLOCK

    with tempfile.TemporaryDirectory(prefix="mxint8_diff_") as tmpdir:
        ours = run_host_ref(binary, a, b, tmpdir)

    examples = {"scale": [], "elements": [], "gemm": []}
    counts = {"scale": 0, "elements": 0, "gemm": 0}
    blocks_total = blocks_zero = 0
    byte_min, byte_max = 255, 0

    # check 1: scale bytes (A blocks along rows; B blocks down K per column,
    # so B is checked transposed).
    for label, vals, scales in (("A", a, ours["a_scale"]),
                                ("B", b.T.copy(), ours["b_scale"].T.copy())):
        for r in range(vals.shape[0]):
            for blk in range(kb):
                blocks_total += 1
                block = vals[r, blk * BLOCK:min((blk + 1) * BLOCK, k)]
                if not np.any(block):
                    blocks_zero += 1
                want = expected_scale_byte(block)
                got = int(scales[r, blk])
                byte_min, byte_max = min(byte_min, got), max(byte_max, got)
                if got != want:
                    counts["scale"] += 1
                    if len(examples["scale"]) < MISMATCH_DETAIL_LIMIT:
                        examples["scale"].append({"tensor": label, "row": r,
                                                  "block": blk, "ours": got,
                                                  "expected": want})

    # check 2: dequantized elements, bit-exact vs microxcaling (binary64).
    ext_a = microxcaling_dequant(a, round_mode)
    ext_b = microxcaling_dequant(b.T.copy(), round_mode).T
    our_a = dequant_rows(ours["a_payload"], ours["a_scale"], k)
    our_b = dequant_rows(ours["b_payload"].T.copy(), ours["b_scale"].T.copy(), k).T
    for label, mine, ext in (("A", our_a, ext_a), ("B", our_b, ext_b)):
        bad = np.argwhere(mine != ext)
        counts["elements"] += int(bad.shape[0])
        for idx in bad[:MISMATCH_DETAIL_LIMIT - len(examples["elements"])]:
            i, j = int(idx[0]), int(idx[1])
            examples["elements"].append({"tensor": label, "i": i, "j": j,
                                         "ours": mine[i, j], "external": ext[i, j]})

    # check 3: golden C vs exact-integer policy GEMM over payloads recovered
    # from microxcaling's dequantized tensors (independent value path; uses
    # our scale bytes, already validated by check 1).
    pa = np.rint(dequant_inverse(ext_a, ours["a_scale"], k)).astype(np.int64)
    pb = np.rint(dequant_inverse(ext_b.T.copy(), ours["b_scale"].T.copy(), k)
                 ).astype(np.int64).T
    payloads_in_range = bool(np.all(np.abs(pa) <= 127) and np.all(np.abs(pb) <= 127))
    c_ext = policy_gemm_exact(pa, pb, ours["a_scale"], ours["b_scale"], k)
    bad = np.argwhere(c_ext != ours["c"])
    counts["gemm"] = int(bad.shape[0])
    for idx in bad[:MISMATCH_DETAIL_LIMIT]:
        i, j = int(idx[0]), int(idx[1])
        examples["gemm"].append({"i": i, "j": j, "golden_c": int(ours["c"][i, j]),
                                 "external_path": int(c_ext[i, j])})

    passed = all(v == 0 for v in counts.values()) and payloads_in_range
    return {
        "seed": int(seed), "m": m, "n": n, "k": k, "k_blocks": kb, "mode": mode,
        "round": round_mode, "blocks_total": blocks_total,
        "blocks_all_zero": blocks_zero,
        "scale_byte_range": [byte_min, byte_max],
        "mismatches": counts,
        "mismatch_examples": examples,
        "external_payloads_in_range": payloads_in_range,
        "pass": bool(passed),
    }


def dequant_inverse(dequant_rows_mat, scale_rows, k):
    """Recover payload-grid values: dequant * 2^(6-e), blockwise (axis 1)."""
    out = np.zeros(dequant_rows_mat.shape, dtype=np.float64)
    kb = (k + BLOCK - 1) // BLOCK
    for blk in range(kb):
        lo, hi = blk * BLOCK, min((blk + 1) * BLOCK, k)
        e = scale_rows[:, blk].astype(np.int64) - 127
        out[:, lo:hi] = dequant_rows_mat[:, lo:hi] \
            * np.exp2((FRAC_BITS - e).astype(np.float64))[:, None]
    return out


def versions():
    import torch
    import mx
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version
    mx_ver = "unknown"
    for dist in ("microxcaling", "mx"):  # pip names the GitHub package "mx"
        try:
            mx_ver = pkg_version(dist)
            break
        except PackageNotFoundError:
            pass
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "microxcaling": mx_ver,
        "microxcaling_note": "installed --no-deps (its torch==2.2.0 pin has no "
                             "wheel for this Python); CPU path only",
        "mx_module": os.path.dirname(mx.__file__),
    }


SWEEP_PROGRAM = [
    # (shape, mode) x seeds 1..10 each: >= 1000 blocks total, half band-targeted.
    ((16, 16, 128), "random"),
    ((16, 16, 128), "band"),
    ((8, 8, 512), "random"),
    ((8, 8, 512), "band"),
    ((4, 4, 40), "random"),  # K-tail (40 = 32 + 8) coverage
    ((4, 4, 40), "band"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="microxcaling", choices=["microxcaling"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--round", default="even", choices=["even"])
    p.add_argument("--mode", default="random", choices=["random", "band"])
    p.add_argument("--sweep", action="store_true",
                   help="run the fixed >=1000-block program (ignores shape/seed/mode)")
    p.add_argument("--json-out", default=None)
    args = p.parse_args()

    binary = build_host_ref()
    cases = ([(s, m, n, k, mode) for ((m, n, k), mode) in SWEEP_PROGRAM
              for s in range(1, 11)]
             if args.sweep else
             [(args.seed, args.m, args.n, args.k, args.mode)])

    results = []
    for seed, m, n, k, mode in cases:
        r = run_case(binary, seed, m, n, k, mode, args.round)
        results.append(r)
        status = "PASS" if r["pass"] else "FAIL"
        print(f"{status} seed={seed} m={m} n={n} k={k} mode={mode} "
              f"blocks={r['blocks_total']} "
              f"mism(scale/elem/gemm)={r['mismatches']['scale']}/"
              f"{r['mismatches']['elements']}/{r['mismatches']['gemm']}")

    report = {
        "backend": args.backend,
        "versions": versions(),
        "blocks_total": sum(r["blocks_total"] for r in results),
        "blocks_band_mode": sum(r["blocks_total"] for r in results
                                if r["mode"] == "band"),
        "pass": all(r["pass"] for r in results),
        "cases": results,
    }
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {args.json_out}")
    print(f"OVERALL: {'PASS' if report['pass'] else 'FAIL'} "
          f"({report['blocks_total']} blocks, "
          f"{report['blocks_band_mode']} in divergence-band mode)")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
