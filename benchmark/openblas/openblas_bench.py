#!/usr/bin/env python3
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import numpy as np


def best(fn, rounds=5):
    fn()
    # best-of-21 floor, symmetric with the C# twin (which also warms JIT to steady-state first): a min
    # over only 3-7 samples of a ~60 us-1.5 ms op is noisy; NumPy is native/low-variance so this only
    # stabilises the reported minimum.
    rounds = max(rounds, 21)
    value = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        value = min(value, (time.perf_counter() - start) * 1000)
    return value


def emit(route, op, n, fn, rounds=5):
    print(f"{route}|{op}|{n}|f64\t{best(fn, rounds):.6f}")


np.random.seed(42)
vector_n = 100_000
matrix_n = 128
vector = np.random.random(vector_n)
kernel = np.random.random(31)
matrix_a = np.random.random((matrix_n, matrix_n))
matrix_b = np.random.random((matrix_n, matrix_n))
matrix_vector = np.random.random(matrix_n)

emit("product", "dot", vector_n, lambda: np.dot(vector, vector))
emit("product", "inner", vector_n, lambda: np.inner(vector, vector))
emit("product", "vdot", vector_n, lambda: np.vdot(vector, vector))
emit("product", "vecdot", vector_n, lambda: np.vecdot(vector, vector))
emit("product", "matmul", matrix_n, lambda: np.matmul(matrix_a, matrix_b))
emit("product", "matvec", matrix_n, lambda: np.matvec(matrix_a, matrix_vector))
emit("product", "vecmat", matrix_n, lambda: np.vecmat(matrix_vector, matrix_a))
emit("sliding", "correlate", vector_n, lambda: np.correlate(vector, kernel, mode="same"))
emit("sliding", "convolve", vector_n, lambda: np.convolve(vector, kernel, mode="same"))

poly_x = np.linspace(-1, 1, 1_000)
poly_y = 3.0 * poly_x * poly_x - 2.0 * poly_x + 1.0
poly_coefficients = np.linspace(1, 2, 17)
emit("composed", "polyfit", 1_000, lambda: np.polyfit(poly_x, poly_y, 3))
emit("composed", "roots", 16, lambda: np.roots(poly_coefficients))

for n in (32, 96):
    np.random.seed(42 + n)
    raw = np.random.random((n, n))
    spd = raw.T @ raw + np.eye(n) * n
    rhs = np.random.random(n)
    rect = np.random.random((2 * n, n))
    rect_rhs = np.random.random(2 * n)
    rounds = 7 if n <= 32 else 3
    emit("lapack", "cholesky", n, lambda: np.linalg.cholesky(spd), rounds)
    emit("lapack", "cond", n, lambda: np.linalg.cond(spd), rounds)
    emit("lapack", "det", n, lambda: np.linalg.det(spd), rounds)
    emit("lapack", "eig", n, lambda: np.linalg.eig(spd), rounds)
    emit("lapack", "eigh", n, lambda: np.linalg.eigh(spd), rounds)
    emit("lapack", "eigvals", n, lambda: np.linalg.eigvals(spd), rounds)
    emit("lapack", "eigvalsh", n, lambda: np.linalg.eigvalsh(spd), rounds)
    emit("lapack", "inv", n, lambda: np.linalg.inv(spd), rounds)
    emit("lapack", "lstsq", n, lambda: np.linalg.lstsq(rect, rect_rhs, rcond=None), rounds)
    emit("lapack", "matrix_rank", n, lambda: np.linalg.matrix_rank(spd), rounds)
    emit("lapack", "pinv", n, lambda: np.linalg.pinv(spd), rounds)
    emit("lapack", "qr", n, lambda: np.linalg.qr(spd), rounds)
    emit("lapack", "slogdet", n, lambda: np.linalg.slogdet(spd), rounds)
    emit("lapack", "solve", n, lambda: np.linalg.solve(spd, rhs), rounds)
    emit("lapack", "svd", n, lambda: np.linalg.svd(spd, full_matrices=False), rounds)
    emit("lapack", "svdvals", n, lambda: np.linalg.svdvals(spd), rounds)
    emit("lapack", "tensorinv", n, lambda: np.linalg.tensorinv(spd, ind=1), rounds)
    emit("lapack", "tensorsolve", n, lambda: np.linalg.tensorsolve(spd, rhs), rounds)
    emit("hybrid", "matrix_power_-1", n, lambda: np.linalg.matrix_power(spd, -1), rounds)
    emit("hybrid", "matrix_norm_2", n, lambda: np.linalg.matrix_norm(spd, ord=2), rounds)
    emit("hybrid", "norm_2", n, lambda: np.linalg.norm(spd, ord=2), rounds)
