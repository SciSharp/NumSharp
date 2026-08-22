#!/usr/bin/env python3
import json
import os
import time

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np


def best(fn, rounds=5):
    fn()
    value = float("inf")
    for _ in range(rounds):
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


np.random.seed(42)
vector_n = 100_000
matrix_n = 128
vector = np.random.random(vector_n)
kernel = np.random.random(31)
matrix_a = np.random.random((matrix_n, matrix_n))
matrix_b = np.random.random((matrix_n, matrix_n))
matrix_vector = np.random.random(matrix_n)

emit("np.dot", "np.dot(a, b)", "openblas_optional", vector_n, lambda: np.dot(vector, vector))
emit("ndarray.dot", "a.dot(b)", "openblas_optional", vector_n, lambda: vector.dot(vector))
emit("np.inner", "np.inner(a, b)", "openblas_optional", vector_n, lambda: np.inner(vector, vector))
emit("np.vdot", "np.vdot(a, b)", "openblas_optional", vector_n, lambda: np.vdot(vector, vector))
emit("np.vecdot", "np.vecdot(a, b)", "openblas_optional", vector_n, lambda: np.vecdot(vector, vector))
emit("np.linalg.vecdot", "np.linalg.vecdot(a, b)", "openblas_optional", vector_n, lambda: np.linalg.vecdot(vector, vector))
emit("np.matmul", "np.matmul(a, b)", "openblas_optional", matrix_n, lambda: np.matmul(matrix_a, matrix_b))
emit("np.linalg.matmul", "np.linalg.matmul(a, b)", "openblas_optional", matrix_n, lambda: np.linalg.matmul(matrix_a, matrix_b))
emit("np.matvec", "np.matvec(a, b)", "openblas_optional", matrix_n, lambda: np.matvec(matrix_a, matrix_vector))
emit("np.vecmat", "np.vecmat(a, b)", "openblas_optional", matrix_n, lambda: np.vecmat(matrix_vector, matrix_a))
emit("np.einsum", "np.einsum('ij,jk->ik', a, b)", "openblas_optional_composed", matrix_n, lambda: np.einsum("ij,jk->ik", matrix_a, matrix_b), "ij,jk->ik")
emit("np.tensordot", "np.tensordot(a, b, axes=1)", "openblas_optional_composed", matrix_n, lambda: np.tensordot(matrix_a, matrix_b, axes=1), "axes=1")
emit("np.linalg.tensordot", "np.linalg.tensordot(a, b, axes=1)", "openblas_optional_composed", matrix_n, lambda: np.linalg.tensordot(matrix_a, matrix_b, axes=1), "axes=1")
emit("np.linalg.multi_dot", "np.linalg.multi_dot(arrays)", "openblas_optional_composed", matrix_n, lambda: np.linalg.multi_dot((matrix_a, matrix_b, matrix_a)))
emit("np.correlate", "np.correlate(a, v, mode='same')", "openblas_optional_sliding_dot", vector_n, lambda: np.correlate(vector, kernel, mode="same"), "mode=same,kernel=31", "Statistics")
emit("np.convolve", "np.convolve(a, v, mode='same')", "openblas_optional_sliding_dot", vector_n, lambda: np.convolve(vector, kernel, mode="same"), "mode=same,kernel=31", "Statistics")

poly_x = np.linspace(-1, 1, 1_000)
poly_y = 3.0 * poly_x * poly_x - 2.0 * poly_x + 1.0
poly_coefficients = np.linspace(1, 2, 17)
emit("np.polyfit", "np.polyfit(x, y, 3)", "openblas_required_composed", 1_000, lambda: np.polyfit(poly_x, poly_y, 3), suite="Api")
emit("np.roots", "np.roots(p)", "openblas_required_composed", 16, lambda: np.roots(poly_coefficients), suite="Api")

for n in (32, 96):
    np.random.seed(42 + n)
    raw = np.random.random((n, n))
    spd = raw.T @ raw + np.eye(n) * n
    rhs = np.random.random(n)
    rect = np.random.random((2 * n, n))
    rect_rhs = np.random.random(2 * n)
    rounds = 7 if n <= 32 else 3
    emit("np.linalg.cholesky", "np.linalg.cholesky(a)", "openblas_required", n, lambda: np.linalg.cholesky(spd), rounds=rounds)
    emit("np.linalg.cond", "np.linalg.cond(a)", "openblas_required_composed", n, lambda: np.linalg.cond(spd), rounds=rounds)
    emit("np.linalg.det", "np.linalg.det(a)", "openblas_required", n, lambda: np.linalg.det(spd), rounds=rounds)
    emit("np.linalg.eig", "np.linalg.eig(a)", "openblas_required", n, lambda: np.linalg.eig(spd), rounds=rounds)
    emit("np.linalg.eigh", "np.linalg.eigh(a)", "openblas_required", n, lambda: np.linalg.eigh(spd), rounds=rounds)
    emit("np.linalg.eigvals", "np.linalg.eigvals(a)", "openblas_required", n, lambda: np.linalg.eigvals(spd), rounds=rounds)
    emit("np.linalg.eigvalsh", "np.linalg.eigvalsh(a)", "openblas_required", n, lambda: np.linalg.eigvalsh(spd), rounds=rounds)
    emit("np.linalg.inv", "np.linalg.inv(a)", "openblas_required", n, lambda: np.linalg.inv(spd), rounds=rounds)
    emit("np.linalg.lstsq", "np.linalg.lstsq(a, b)", "openblas_required", n, lambda: np.linalg.lstsq(rect, rect_rhs, rcond=None), rounds=rounds)
    emit("np.linalg.matrix_rank", "np.linalg.matrix_rank(a)", "openblas_required_composed", n, lambda: np.linalg.matrix_rank(spd), rounds=rounds)
    emit("np.linalg.pinv", "np.linalg.pinv(a)", "openblas_required_composed", n, lambda: np.linalg.pinv(spd), rounds=rounds)
    emit("np.linalg.qr", "np.linalg.qr(a)", "openblas_required", n, lambda: np.linalg.qr(spd), rounds=rounds)
    emit("np.linalg.slogdet", "np.linalg.slogdet(a)", "openblas_required", n, lambda: np.linalg.slogdet(spd), rounds=rounds)
    emit("np.linalg.solve", "np.linalg.solve(a, b)", "openblas_required", n, lambda: np.linalg.solve(spd, rhs), rounds=rounds)
    emit("np.linalg.svd", "np.linalg.svd(a)", "openblas_required", n, lambda: np.linalg.svd(spd, full_matrices=False), rounds=rounds)
    emit("np.linalg.svdvals", "np.linalg.svdvals(a)", "openblas_required", n, lambda: np.linalg.svdvals(spd), rounds=rounds)
    emit("np.linalg.tensorinv", "np.linalg.tensorinv(a)", "openblas_required_composed", n, lambda: np.linalg.tensorinv(spd, ind=1), rounds=rounds)
    emit("np.linalg.tensorsolve", "np.linalg.tensorsolve(a, b)", "openblas_required_composed", n, lambda: np.linalg.tensorsolve(spd, rhs), rounds=rounds)
    emit("np.linalg.matrix_power", "np.linalg.matrix_power(a, -1)", "openblas_hybrid", n, lambda: np.linalg.matrix_power(spd, -1), "n=-1", rounds=rounds)
    emit("np.linalg.matrix_norm", "np.linalg.matrix_norm(a, ord=2)", "openblas_hybrid", n, lambda: np.linalg.matrix_norm(spd, ord=2), "ord=2", rounds=rounds)
    emit("np.linalg.norm", "np.linalg.norm(a, ord=2)", "openblas_hybrid", n, lambda: np.linalg.norm(spd, ord=2), "ord=2", rounds=rounds)
