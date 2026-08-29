#!/usr/bin/env python3
import json
import math
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np


def best(fn, rounds=5):
    fn()
    # Min-time policy, symmetric with the C# twin (which also warms JIT to steady-state first): the
    # reported number is the min over a time-budgeted sweep. A call >20 ms/call runs EXACTLY 100 times;
    # everything else runs enough single-call rounds to span ~200 ms total. The first timed round
    # doubles as the pilot. NumPy is native and low-variance, so this only stabilises the reported
    # minimum, it does not shift it.
    start = time.perf_counter()
    fn()
    value = (time.perf_counter() - start) * 1000
    rounds = 100 if value > 20.0 else min(100_000, max(2, int(math.ceil(200.0 / max(value, 1e-9)))))
    for _ in range(rounds - 1):
        start = time.perf_counter()
        fn()
        value = min(value, (time.perf_counter() - start) * 1000)
    return value


def emit(api, operation, route, n, fn, scenario="", suite="LinearAlgebra", rounds=5):
    print(json.dumps({
        "key": f"{api}|{scenario}|{n}|float64",
        "api": api,
        "operation": operation,
        "suite": suite,
        "category": route,
        "backend_route": route,
        "dtype": "float64",
        "n": n,
        "scenario": scenario,
        "numpy_ms": best(fn, rounds),
    }))


# Every product below reaches an IBlasBackend member directly or reaches dot/matmul through a
# composition. Run ONE shared tier matrix in both profiles; a one-off representative case makes the
# dashboard compare unlike scenario sets and prevents its per-cell fastest-backend selection.
product_tiers = (1_000, 100_000, 10_000_000)
for n in product_tiers:
    np.random.seed(42 + n)
    vector = np.random.random(n)

    # The official LinearAlgebra suite maps the shared workload tier N to a bounded square side.
    side = min(math.isqrt(n), 384)
    matrix_a = np.random.random((side, side))
    matrix_b = np.random.random((side, side))
    matrix_vector = np.random.random(side)

    # LinalgApi's cubic compositions are capped at 128. Keep those exact operands at the shared N
    # keys so 1K/100K join the main matrix instead of cross-pairing different physical shapes.
    linalg_side = min(math.isqrt(n), 128)
    linalg_a = np.random.random((linalg_side, linalg_side))
    linalg_b = np.random.random((linalg_side, linalg_side))

    # np.inner / ndarray.dot join the op-matrix Statistics/NDArray suites, which are memory-heavy
    # families: their published 10M tier runs MEMORY_HEAVY_LARGE_WORKLOAD (1M) physical elements on
    # two DISTINCT arrays. Measure the SAME physical problem — a raw-10M same-vector dot here is 10x
    # the work at half the memory traffic, and combine() rebases this row onto the main matrix's 1M
    # two-array NumPy baseline (the exact cross-pairing the matmul side already fixed).
    work_n = 1_000_000 if n == 10_000_000 else n
    capped_a = np.random.random(work_n) * 100 - 50
    capped_b = np.random.random(work_n) * 100 - 50

    emit("np.dot", "np.dot(a, b)", "openblas_optional", n, lambda: np.dot(vector, vector))
    emit("ndarray.dot", "a.dot(b) [ndarray]", "openblas_optional", n, lambda: capped_a.dot(capped_b), suite="NDArray")
    emit("np.inner", "np.inner(a, b)", "openblas_optional", n, lambda: np.inner(capped_a, capped_b), suite="Statistics")
    emit("np.vdot", "np.vdot(a, b)", "openblas_optional", n, lambda: np.vdot(vector, vector))
    emit("np.vecdot", "np.vecdot(a, b)", "openblas_optional", n, lambda: np.vecdot(vector, vector))
    emit("np.linalg.vecdot", "np.linalg.vecdot(a, b)", "openblas_optional", n, lambda: np.linalg.vecdot(vector, vector))

    emit("np.matmul", "np.matmul(a, b)", "openblas_optional", n, lambda: np.matmul(matrix_a, matrix_b))
    emit("np.linalg.matmul", "np.linalg.matmul(a, b)", "openblas_optional", n, lambda: np.linalg.matmul(matrix_a, matrix_b))
    emit("np.matvec", "np.matvec(a, b)", "openblas_optional", n, lambda: np.matvec(matrix_a, matrix_vector))
    emit("np.vecmat", "np.vecmat(a, b)", "openblas_optional", n, lambda: np.vecmat(matrix_vector, matrix_a))

    # Cover the exact op-matrix einsum/tensordot cells AND a matrix-contraction scenario. Both are
    # product compositions and therefore backend-sensitive even though they have no Try* seam of
    # their own.
    emit("np.einsum", "np.einsum(subscripts, operands)", "openblas_optional_composed", n, lambda: np.einsum("i,i->", vector, vector))
    emit("np.einsum", "np.einsum('ij,jk->ik', a, b)", "openblas_optional_composed", n, lambda: np.einsum("ij,jk->ik", matrix_a, matrix_b), "ij,jk->ik")
    emit("np.tensordot", "np.tensordot(a, b)", "openblas_optional_composed", n, lambda: np.tensordot(vector, vector, axes=1))
    emit("np.tensordot", "np.tensordot(a, b, axes=1)", "openblas_optional_composed", n, lambda: np.tensordot(matrix_a, matrix_b, axes=1), "axes=1,matrix")
    emit("np.linalg.tensordot", "np.linalg.tensordot(a, b)", "openblas_optional_composed", n, lambda: np.linalg.tensordot(linalg_a, linalg_b, axes=1))
    emit("np.linalg.multi_dot", "np.linalg.multi_dot(arrays)", "openblas_optional_composed", n, lambda: np.linalg.multi_dot((linalg_a, linalg_b, linalg_a)))
    emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, 3)", "openblas_optional_composed", n, lambda: np.linalg.matrix_power(linalg_a, 3))

# Sliding dot is backend-sensitive too and follows the same published workload tiers. The op-matrix
# twin (the Statistics Signal suite) is a memory-heavy family, so its published 10M tier runs 1M
# physical signal elements — match it or combine() pairs a 10M-position sweep against a 1M baseline.
for n in product_tiers:
    np.random.seed(84 + n)
    work_n = 1_000_000 if n == 10_000_000 else n
    signal = np.random.random(work_n) * 100 - 50
    kernel = np.random.random(31) * 2 - 1
    emit("np.correlate", "np.correlate(a, kernel)", "openblas_optional_sliding_dot", n, lambda: np.correlate(signal, kernel, mode="same"), suite="Statistics")
    emit("np.convolve", "np.convolve(a, kernel)", "openblas_optional_sliding_dot", n, lambda: np.convolve(signal, kernel, mode="same"), suite="Statistics")

# API compositions also publish only the shared workload tiers. Their physical problems grow on a
# bounded scale (samples 1K/10K/100K; polynomial degrees 16/32/64) rather than pretending a degree-
# ten-million companion matrix is a runnable benchmark.
for tier_n, sample_n, coefficient_count in (
    (1_000, 1_000, 17),
    (100_000, 10_000, 33),
    (10_000_000, 100_000, 65),
):
    poly_x = np.linspace(-1, 1, sample_n)
    poly_y = 3.0 * poly_x * poly_x - 2.0 * poly_x + 1.0
    poly_coefficients = np.linspace(1, 2, coefficient_count)
    emit("np.polyfit", "np.polyfit(x, y, 3)", "openblas_required_composed", tier_n, lambda: np.polyfit(poly_x, poly_y, 3), suite="Api")
    emit("np.roots", "np.roots(p)", "openblas_required_composed", tier_n, lambda: np.roots(poly_coefficients), suite="Api")

# LAPACK's N is the SAME published workload tier as every other profile row. The physical matrix
# side is bounded per tier: 1K -> 32, 100K -> 96, 10M -> 128. This preserves the useful scaling
# sweep without exposing implementation-specific side lengths as dashboard size tiers.
for tier_n, matrix_n in ((1_000, 32), (100_000, 96), (10_000_000, 128)):
    np.random.seed(42 + tier_n)
    raw = np.random.random((matrix_n, matrix_n))
    spd = raw.T @ raw + np.eye(matrix_n) * matrix_n
    rhs = np.random.random(matrix_n)
    rect = np.random.random((2 * matrix_n, matrix_n))
    rect_rhs = np.random.random(2 * matrix_n)
    rounds = 7 if matrix_n <= 32 else 3
    emit("np.linalg.cholesky", "np.linalg.cholesky(a)", "openblas_required", tier_n, lambda: np.linalg.cholesky(spd), rounds=rounds)
    emit("np.linalg.cond", "np.linalg.cond(a)", "openblas_required_composed", tier_n, lambda: np.linalg.cond(spd), rounds=rounds)
    emit("np.linalg.det", "np.linalg.det(a)", "openblas_required", tier_n, lambda: np.linalg.det(spd), rounds=rounds)
    emit("np.linalg.eig", "np.linalg.eig(a)", "openblas_required", tier_n, lambda: np.linalg.eig(spd), rounds=rounds)
    emit("np.linalg.eigh", "np.linalg.eigh(a)", "openblas_required", tier_n, lambda: np.linalg.eigh(spd), rounds=rounds)
    emit("np.linalg.eigvals", "np.linalg.eigvals(a)", "openblas_required", tier_n, lambda: np.linalg.eigvals(spd), rounds=rounds)
    emit("np.linalg.eigvalsh", "np.linalg.eigvalsh(a)", "openblas_required", tier_n, lambda: np.linalg.eigvalsh(spd), rounds=rounds)
    emit("np.linalg.inv", "np.linalg.inv(a)", "openblas_required", tier_n, lambda: np.linalg.inv(spd), rounds=rounds)
    emit("np.linalg.lstsq", "np.linalg.lstsq(a, b)", "openblas_required", tier_n, lambda: np.linalg.lstsq(rect, rect_rhs, rcond=None), rounds=rounds)
    emit("np.linalg.matrix_rank", "np.linalg.matrix_rank(a)", "openblas_required_composed", tier_n, lambda: np.linalg.matrix_rank(spd), rounds=rounds)
    emit("np.linalg.pinv", "np.linalg.pinv(a)", "openblas_required_composed", tier_n, lambda: np.linalg.pinv(spd), rounds=rounds)
    emit("np.linalg.qr", "np.linalg.qr(a)", "openblas_required", tier_n, lambda: np.linalg.qr(spd), rounds=rounds)
    emit("np.linalg.slogdet", "np.linalg.slogdet(a)", "openblas_required", tier_n, lambda: np.linalg.slogdet(spd), rounds=rounds)
    emit("np.linalg.solve", "np.linalg.solve(a, b)", "openblas_required", tier_n, lambda: np.linalg.solve(spd, rhs), rounds=rounds)
    emit("np.linalg.svd", "np.linalg.svd(a)", "openblas_required", tier_n, lambda: np.linalg.svd(spd, full_matrices=False), rounds=rounds)
    emit("np.linalg.svdvals", "np.linalg.svdvals(a)", "openblas_required", tier_n, lambda: np.linalg.svdvals(spd), rounds=rounds)
    emit("np.linalg.tensorinv", "np.linalg.tensorinv(a)", "openblas_required_composed", tier_n, lambda: np.linalg.tensorinv(spd, ind=1), rounds=rounds)
    emit("np.linalg.tensorsolve", "np.linalg.tensorsolve(a, b)", "openblas_required_composed", tier_n, lambda: np.linalg.tensorsolve(spd, rhs), rounds=rounds)
    emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, -1)", "openblas_hybrid", tier_n, lambda: np.linalg.matrix_power(spd, -1), "n=-1", rounds=rounds)
    emit("np.linalg.matrix_norm", "np.linalg.matrix_norm(a, ord=2)", "openblas_hybrid", tier_n, lambda: np.linalg.matrix_norm(spd, ord=2), "ord=2", rounds=rounds)
    emit("np.linalg.norm", "np.linalg.norm(a, ord=2)", "openblas_hybrid", tier_n, lambda: np.linalg.norm(spd, ord=2), "ord=2", rounds=rounds)
