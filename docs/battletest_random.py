#!/usr/bin/env python3
"""
NumPy Random Battletest - Penetration-level testing of ALL np.random methods
================================================================================
This script exhaustively tests EVERY np.random method with ALL edge cases including:
- Parameter boundaries (0, negative, inf, nan, very large, very small)
- Size parameter variations (None, int, tuple, empty tuple, 0-sized)
- Return types (scalar vs array), dtypes, shapes
- Error conditions with exact error messages/types
- Special numbers (inf, -inf, nan)
- Seed reproducibility
- State save/restore

Run with: python battletest_random.py > battletest_random_output.txt 2>&1
"""

import numpy as np
import sys
import traceback
from contextlib import contextmanager

# Use legacy RandomState for NumSharp compatibility
rng = np.random.RandomState()

def section(name):
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}\n")

def subsection(name):
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}\n")

def test(description, func):
    """Execute a test and capture result or error"""
    try:
        result = func()
        if isinstance(result, np.ndarray):
            print(f"[OK] {description}")
            print(f"     type={type(result).__name__}, dtype={result.dtype}, shape={result.shape}, ndim={result.ndim}")
            if result.size <= 20:
                print(f"     value={result}")
            elif result.size <= 100:
                print(f"     flat[:20]={result.flat[:20]}")
            else:
                print(f"     first 5 elements={result.flat[:5]}")
        else:
            print(f"[OK] {description}")
            print(f"     type={type(result).__name__}, value={result}")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[ERR] {description}")
        print(f"      {error_type}: {error_msg}")

def test_error_expected(description, func, expected_error_type=None):
    """Execute a test expecting an error"""
    try:
        result = func()
        print(f"[UNEXPECTED OK] {description}")
        if isinstance(result, np.ndarray):
            print(f"     type={type(result).__name__}, dtype={result.dtype}, shape={result.shape}")
        else:
            print(f"     type={type(result).__name__}, value={result}")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        if expected_error_type and error_type != expected_error_type:
            print(f"[WRONG ERR] {description}")
            print(f"      Expected {expected_error_type}, got {error_type}: {error_msg}")
        else:
            print(f"[ERR OK] {description}")
            print(f"      {error_type}: {error_msg}")

def test_seeded(description, seed_val, func):
    """Test with explicit seed for reproducibility verification"""
    try:
        np.random.seed(seed_val)
        result = func()
        if isinstance(result, np.ndarray):
            print(f"[SEEDED] {description} (seed={seed_val})")
            print(f"     type={type(result).__name__}, dtype={result.dtype}, shape={result.shape}")
            if result.size <= 20:
                print(f"     value={result}")
            else:
                print(f"     flat[:10]={result.flat[:10]}")
        else:
            print(f"[SEEDED] {description} (seed={seed_val})")
            print(f"     type={type(result).__name__}, value={result}")
    except Exception as e:
        print(f"[SEEDED ERR] {description} (seed={seed_val})")
        print(f"      {type(e).__name__}: {str(e)}")

# Special values for edge case testing
INF = float('inf')
NEG_INF = float('-inf')
NAN = float('nan')
VERY_LARGE = 1e308
VERY_SMALL = 1e-308
EPSILON = np.finfo(float).eps

# ============================================================================
#  SEED TESTING
# ============================================================================
section("SEED")

subsection("seed() - Valid Seeds")
test("seed(0)", lambda: (np.random.seed(0), np.random.random())[1])
test("seed(1)", lambda: (np.random.seed(1), np.random.random())[1])
test("seed(42)", lambda: (np.random.seed(42), np.random.random())[1])
test("seed(2**31-1)", lambda: (np.random.seed(2**31-1), np.random.random())[1])
test("seed(2**32-1)", lambda: (np.random.seed(2**32-1), np.random.random())[1])

subsection("seed() - Invalid Seeds")
test_error_expected("seed(-1)", lambda: np.random.seed(-1), "ValueError")
test_error_expected("seed(-2**31)", lambda: np.random.seed(-2**31), "ValueError")
test_error_expected("seed(2**32)", lambda: np.random.seed(2**32), "ValueError")
test_error_expected("seed(2**33)", lambda: np.random.seed(2**33), "ValueError")
test_error_expected("seed(2**64)", lambda: np.random.seed(2**64), "ValueError")

subsection("seed() - Type Acceptance")
test("seed(np.int32(42))", lambda: (np.random.seed(np.int32(42)), np.random.random())[1])
test("seed(np.int64(42))", lambda: (np.random.seed(np.int64(42)), np.random.random())[1])
test("seed(np.uint32(42))", lambda: (np.random.seed(np.uint32(42)), np.random.random())[1])
test("seed(np.uint64(42))", lambda: (np.random.seed(np.uint64(42)), np.random.random())[1])
test_error_expected("seed(42.0)", lambda: np.random.seed(42.0), "TypeError")
test_error_expected("seed(42.5)", lambda: np.random.seed(42.5), "TypeError")
test_error_expected("seed('42')", lambda: np.random.seed('42'), "TypeError")
test_error_expected("seed(None)", lambda: np.random.seed(None))  # Actually OK in NumPy!

subsection("seed() - Array Seeds")
test("seed([1,2,3,4])", lambda: (np.random.seed([1,2,3,4]), np.random.random())[1])
test("seed(np.array([1,2,3]))", lambda: (np.random.seed(np.array([1,2,3])), np.random.random())[1])
test("seed([])", lambda: (np.random.seed([]), np.random.random())[1])
test_error_expected("seed([[1,2],[3,4]]) 2D", lambda: np.random.seed([[1,2],[3,4]]))

subsection("seed() - Reproducibility Verification")
def test_reproducibility():
    np.random.seed(12345)
    a = np.random.random(10)
    np.random.seed(12345)
    b = np.random.random(10)
    return np.array_equal(a, b), a, b
test("Reproducibility check", test_reproducibility)

# ============================================================================
#  STATE MANAGEMENT
# ============================================================================
section("STATE MANAGEMENT")

subsection("get_state() / set_state()")
def test_state():
    np.random.seed(42)
    state = np.random.get_state()
    print(f"     State type: {type(state)}")
    print(f"     State[0] (algorithm): {state[0]}")
    print(f"     State[1] shape (key): {state[1].shape}, dtype={state[1].dtype}")
    print(f"     State[2] (pos): {state[2]}")
    print(f"     State[3] (has_gauss): {state[3]}")
    print(f"     State[4] (cached_gaussian): {state[4]}")
    return state
test("get_state() structure", test_state)

def test_state_restore():
    np.random.seed(42)
    _ = np.random.random(5)  # Consume some randoms
    state = np.random.get_state()
    a = np.random.random(10)
    np.random.set_state(state)
    b = np.random.random(10)
    return np.array_equal(a, b), a, b
test("set_state() restore", test_state_restore)

# ============================================================================
#  RAND
# ============================================================================
section("RAND")

subsection("rand() - Size Variations")
test("rand() - no args", lambda: np.random.rand())
test("rand(1)", lambda: np.random.rand(1))
test("rand(5)", lambda: np.random.rand(5))
test("rand(2,3)", lambda: np.random.rand(2,3))
test("rand(2,3,4)", lambda: np.random.rand(2,3,4))
test("rand(0)", lambda: np.random.rand(0))
test("rand(0,5)", lambda: np.random.rand(0,5))
test("rand(5,0)", lambda: np.random.rand(5,0))
test("rand(1,1,1,1,1)", lambda: np.random.rand(1,1,1,1,1))
test_error_expected("rand(-1)", lambda: np.random.rand(-1), "ValueError")
test_error_expected("rand(2,-3)", lambda: np.random.rand(2,-3), "ValueError")

subsection("rand() - Output Properties")
test_seeded("rand(1000) bounds check", 42, lambda: (np.random.rand(1000).min(), np.random.rand(1000).max()))

# ============================================================================
#  RANDN
# ============================================================================
section("RANDN")

subsection("randn() - Size Variations")
test("randn() - no args", lambda: np.random.randn())
test("randn(1)", lambda: np.random.randn(1))
test("randn(5)", lambda: np.random.randn(5))
test("randn(2,3)", lambda: np.random.randn(2,3))
test("randn(2,3,4)", lambda: np.random.randn(2,3,4))
test("randn(0)", lambda: np.random.randn(0))
test_error_expected("randn(-1)", lambda: np.random.randn(-1), "ValueError")

subsection("randn() - Seeded Values")
test_seeded("randn(10)", 42, lambda: np.random.randn(10))
test_seeded("randn(3,3)", 42, lambda: np.random.randn(3,3))

# ============================================================================
#  RANDINT
# ============================================================================
section("RANDINT")

subsection("randint() - Basic Usage")
test("randint(10)", lambda: np.random.randint(10))
test("randint(0, 10)", lambda: np.random.randint(0, 10))
test("randint(5, 10)", lambda: np.random.randint(5, 10))
test("randint(-10, 10)", lambda: np.random.randint(-10, 10))
test("randint(-10, -5)", lambda: np.random.randint(-10, -5))

subsection("randint() - Size Parameter")
test("randint(10, size=None)", lambda: np.random.randint(10, size=None))
test("randint(10, size=5)", lambda: np.random.randint(10, size=5))
test("randint(10, size=(2,3))", lambda: np.random.randint(10, size=(2,3)))
test("randint(10, size=(2,3,4))", lambda: np.random.randint(10, size=(2,3,4)))
test("randint(10, size=())", lambda: np.random.randint(10, size=()))
test("randint(10, size=(0,))", lambda: np.random.randint(10, size=(0,)))
test("randint(10, size=(5,0))", lambda: np.random.randint(10, size=(5,0)))

subsection("randint() - dtype Parameter")
test("randint(10, dtype=np.int8)", lambda: np.random.randint(10, size=5, dtype=np.int8))
test("randint(10, dtype=np.int16)", lambda: np.random.randint(10, size=5, dtype=np.int16))
test("randint(10, dtype=np.int32)", lambda: np.random.randint(10, size=5, dtype=np.int32))
test("randint(10, dtype=np.int64)", lambda: np.random.randint(10, size=5, dtype=np.int64))
test("randint(10, dtype=np.uint8)", lambda: np.random.randint(10, size=5, dtype=np.uint8))
test("randint(10, dtype=np.uint16)", lambda: np.random.randint(10, size=5, dtype=np.uint16))
test("randint(10, dtype=np.uint32)", lambda: np.random.randint(10, size=5, dtype=np.uint32))
test("randint(10, dtype=np.uint64)", lambda: np.random.randint(10, size=5, dtype=np.uint64))
test("randint(10, dtype=bool)", lambda: np.random.randint(2, size=5, dtype=bool))

subsection("randint() - Boundary Values")
test("randint(0, 1)", lambda: np.random.randint(0, 1))
test("randint(0, 1, size=10)", lambda: np.random.randint(0, 1, size=10))
test("randint(-128, 127, dtype=np.int8)", lambda: np.random.randint(-128, 127, size=5, dtype=np.int8))
test("randint(0, 255, dtype=np.uint8)", lambda: np.random.randint(0, 255, size=5, dtype=np.uint8))
test("randint(0, 256, dtype=np.uint8)", lambda: np.random.randint(0, 256, size=5, dtype=np.uint8))
test("randint(-2**31, 2**31-1, dtype=np.int32)", lambda: np.random.randint(-2**31, 2**31-1, size=5, dtype=np.int32))
test("randint(0, 2**32-1, dtype=np.uint32)", lambda: np.random.randint(0, 2**32-1, size=5, dtype=np.uint32))
test("randint(0, 2**32, dtype=np.uint32)", lambda: np.random.randint(0, 2**32, size=5, dtype=np.uint32))
test("randint(-2**63, 2**63-1, dtype=np.int64)", lambda: np.random.randint(-2**63, 2**63-1, size=5, dtype=np.int64))
test("randint(0, 2**64-1, dtype=np.uint64)", lambda: np.random.randint(0, 2**64-1, size=5, dtype=np.uint64))

subsection("randint() - Errors")
test_error_expected("randint(0)", lambda: np.random.randint(0), "ValueError")
test_error_expected("randint(10, 5) low>high", lambda: np.random.randint(10, 5), "ValueError")
test_error_expected("randint(5, 5) low==high", lambda: np.random.randint(5, 5), "ValueError")
test_error_expected("randint(-1, size=-1)", lambda: np.random.randint(10, size=-1), "ValueError")
test_error_expected("randint(256, dtype=np.int8) overflow", lambda: np.random.randint(256, size=5, dtype=np.int8))
test_error_expected("randint(-1, 10, dtype=np.uint8) negative with uint", lambda: np.random.randint(-1, 10, size=5, dtype=np.uint8))
test_error_expected("randint(0, 2**32+1, dtype=np.uint32)", lambda: np.random.randint(0, 2**32+1, size=5, dtype=np.uint32))

subsection("randint() - Seeded Values")
test_seeded("randint(100, size=5)", 42, lambda: np.random.randint(100, size=5))
test_seeded("randint(0, 100, size=5)", 42, lambda: np.random.randint(0, 100, size=5))
test_seeded("randint(-50, 50, size=5)", 42, lambda: np.random.randint(-50, 50, size=5))

# ============================================================================
#  RANDOM / RANDOM_SAMPLE
# ============================================================================
section("RANDOM / RANDOM_SAMPLE")

subsection("random_sample() - Size Variations")
test("random_sample()", lambda: np.random.random_sample())
test("random_sample(None)", lambda: np.random.random_sample(None))
test("random_sample(5)", lambda: np.random.random_sample(5))
test("random_sample((2,3))", lambda: np.random.random_sample((2,3)))
test("random_sample((0,))", lambda: np.random.random_sample((0,)))
test_error_expected("random_sample(-1)", lambda: np.random.random_sample(-1), "ValueError")

subsection("random() - Alias")
test("random()", lambda: np.random.random())
test("random(5)", lambda: np.random.random(5))
test("random((2,3))", lambda: np.random.random((2,3)))

# ============================================================================
#  UNIFORM
# ============================================================================
section("UNIFORM")

subsection("uniform() - Basic Usage")
test("uniform()", lambda: np.random.uniform())
test("uniform(0, 1)", lambda: np.random.uniform(0, 1))
test("uniform(-1, 1)", lambda: np.random.uniform(-1, 1))
test("uniform(10, 20)", lambda: np.random.uniform(10, 20))
test("uniform(0, 1, size=5)", lambda: np.random.uniform(0, 1, size=5))
test("uniform(0, 1, size=(2,3))", lambda: np.random.uniform(0, 1, size=(2,3)))

subsection("uniform() - Edge Cases")
test("uniform(0, 0)", lambda: np.random.uniform(0, 0, size=5))
test("uniform(5, 5)", lambda: np.random.uniform(5, 5, size=5))
test("uniform(10, 5) low>high", lambda: np.random.uniform(10, 5, size=5))  # NumPy allows this!
test("uniform(-inf, inf)", lambda: np.random.uniform(-1e308, 1e308, size=5))
test("uniform(0, VERY_LARGE)", lambda: np.random.uniform(0, 1e308, size=5))
test("uniform(VERY_SMALL, 1)", lambda: np.random.uniform(1e-308, 1, size=5))

subsection("uniform() - Special Values")
test_error_expected("uniform(nan, 1)", lambda: np.random.uniform(float('nan'), 1, size=5))
test_error_expected("uniform(0, nan)", lambda: np.random.uniform(0, float('nan'), size=5))
test_error_expected("uniform(inf, inf)", lambda: np.random.uniform(float('inf'), float('inf'), size=5))
test_error_expected("uniform(-inf, -inf)", lambda: np.random.uniform(float('-inf'), float('-inf'), size=5))

subsection("uniform() - Seeded")
test_seeded("uniform(0, 100, size=5)", 42, lambda: np.random.uniform(0, 100, size=5))

# ============================================================================
#  NORMAL
# ============================================================================
section("NORMAL")

subsection("normal() - Basic Usage")
test("normal()", lambda: np.random.normal())
test("normal(0, 1)", lambda: np.random.normal(0, 1))
test("normal(10, 2)", lambda: np.random.normal(10, 2))
test("normal(-5, 0.5)", lambda: np.random.normal(-5, 0.5))
test("normal(0, 1, size=5)", lambda: np.random.normal(0, 1, size=5))
test("normal(0, 1, size=(2,3))", lambda: np.random.normal(0, 1, size=(2,3)))

subsection("normal() - Edge Cases")
test("normal(0, 0)", lambda: np.random.normal(0, 0, size=5))  # All zeros
test("normal(1e308, 1)", lambda: np.random.normal(1e308, 1, size=5))
test("normal(0, 1e308)", lambda: np.random.normal(0, 1e308, size=5))
test("normal(0, EPSILON)", lambda: np.random.normal(0, np.finfo(float).eps, size=5))

subsection("normal() - Errors")
test_error_expected("normal(0, -1) negative scale", lambda: np.random.normal(0, -1, size=5), "ValueError")
test_error_expected("normal(nan, 1)", lambda: np.random.normal(float('nan'), 1, size=5))
test_error_expected("normal(0, nan)", lambda: np.random.normal(0, float('nan'), size=5))
test_error_expected("normal(0, inf)", lambda: np.random.normal(0, float('inf'), size=5))

subsection("normal() - Seeded")
test_seeded("normal(0, 1, size=10)", 42, lambda: np.random.normal(0, 1, size=10))

# ============================================================================
#  STANDARD_NORMAL
# ============================================================================
section("STANDARD_NORMAL")

subsection("standard_normal() - Size Variations")
test("standard_normal()", lambda: np.random.standard_normal())
test("standard_normal(None)", lambda: np.random.standard_normal(None))
test("standard_normal(5)", lambda: np.random.standard_normal(5))
test("standard_normal((2,3))", lambda: np.random.standard_normal((2,3)))
test("standard_normal((0,))", lambda: np.random.standard_normal((0,)))
test_error_expected("standard_normal(-1)", lambda: np.random.standard_normal(-1), "ValueError")

subsection("standard_normal() - Seeded")
test_seeded("standard_normal(10)", 42, lambda: np.random.standard_normal(10))

# ============================================================================
#  BETA
# ============================================================================
section("BETA")

subsection("beta() - Basic Usage")
test("beta(1, 1)", lambda: np.random.beta(1, 1))
test("beta(0.5, 0.5)", lambda: np.random.beta(0.5, 0.5))
test("beta(2, 5)", lambda: np.random.beta(2, 5))
test("beta(0.1, 0.1)", lambda: np.random.beta(0.1, 0.1))
test("beta(100, 100)", lambda: np.random.beta(100, 100))
test("beta(1, 1, size=5)", lambda: np.random.beta(1, 1, size=5))
test("beta(1, 1, size=(2,3))", lambda: np.random.beta(1, 1, size=(2,3)))

subsection("beta() - Edge Cases")
test("beta(EPSILON, 1)", lambda: np.random.beta(np.finfo(float).eps, 1, size=5))
test("beta(1, EPSILON)", lambda: np.random.beta(1, np.finfo(float).eps, size=5))
test("beta(1e-10, 1e-10)", lambda: np.random.beta(1e-10, 1e-10, size=5))
test("beta(1e10, 1e10)", lambda: np.random.beta(1e10, 1e10, size=5))

subsection("beta() - Errors")
test_error_expected("beta(0, 1)", lambda: np.random.beta(0, 1, size=5), "ValueError")
test_error_expected("beta(1, 0)", lambda: np.random.beta(1, 0, size=5), "ValueError")
test_error_expected("beta(-1, 1)", lambda: np.random.beta(-1, 1, size=5), "ValueError")
test_error_expected("beta(1, -1)", lambda: np.random.beta(1, -1, size=5), "ValueError")
test_error_expected("beta(nan, 1)", lambda: np.random.beta(float('nan'), 1, size=5))
test_error_expected("beta(inf, 1)", lambda: np.random.beta(float('inf'), 1, size=5))

subsection("beta() - Seeded")
test_seeded("beta(2, 5, size=10)", 42, lambda: np.random.beta(2, 5, size=10))

# ============================================================================
#  GAMMA
# ============================================================================
section("GAMMA")

subsection("gamma() - Basic Usage")
test("gamma(1)", lambda: np.random.gamma(1))
test("gamma(1, 1)", lambda: np.random.gamma(1, 1))
test("gamma(0.5, 1)", lambda: np.random.gamma(0.5, 1))
test("gamma(2, 2)", lambda: np.random.gamma(2, 2))
test("gamma(0.1, 1)", lambda: np.random.gamma(0.1, 1))
test("gamma(100, 0.01)", lambda: np.random.gamma(100, 0.01))
test("gamma(1, 1, size=5)", lambda: np.random.gamma(1, 1, size=5))
test("gamma(1, 1, size=(2,3))", lambda: np.random.gamma(1, 1, size=(2,3)))

subsection("gamma() - Edge Cases")
test("gamma(EPSILON, 1)", lambda: np.random.gamma(np.finfo(float).eps, 1, size=5))
test("gamma(1, EPSILON)", lambda: np.random.gamma(1, np.finfo(float).eps, size=5))
test("gamma(1e-10, 1)", lambda: np.random.gamma(1e-10, 1, size=5))
test("gamma(1e10, 1)", lambda: np.random.gamma(1e10, 1, size=5))
test("gamma(1, 1e10)", lambda: np.random.gamma(1, 1e10, size=5))

subsection("gamma() - Errors")
test_error_expected("gamma(0, 1)", lambda: np.random.gamma(0, 1, size=5), "ValueError")
test_error_expected("gamma(-1, 1)", lambda: np.random.gamma(-1, 1, size=5), "ValueError")
test_error_expected("gamma(1, 0)", lambda: np.random.gamma(1, 0, size=5), "ValueError")
test_error_expected("gamma(1, -1)", lambda: np.random.gamma(1, -1, size=5), "ValueError")
test_error_expected("gamma(nan, 1)", lambda: np.random.gamma(float('nan'), 1, size=5))
test_error_expected("gamma(inf, 1)", lambda: np.random.gamma(float('inf'), 1, size=5))

subsection("gamma() - Seeded")
test_seeded("gamma(2, 1, size=10)", 42, lambda: np.random.gamma(2, 1, size=10))

# ============================================================================
#  STANDARD_GAMMA
# ============================================================================
section("STANDARD_GAMMA")

subsection("standard_gamma() - Basic Usage")
test("standard_gamma(1)", lambda: np.random.standard_gamma(1))
test("standard_gamma(0.5)", lambda: np.random.standard_gamma(0.5))
test("standard_gamma(2)", lambda: np.random.standard_gamma(2))
test("standard_gamma(1, size=5)", lambda: np.random.standard_gamma(1, size=5))
test("standard_gamma(1, size=(2,3))", lambda: np.random.standard_gamma(1, size=(2,3)))

subsection("standard_gamma() - Edge Cases")
test("standard_gamma(EPSILON)", lambda: np.random.standard_gamma(np.finfo(float).eps, size=5))
test("standard_gamma(1e-10)", lambda: np.random.standard_gamma(1e-10, size=5))
test("standard_gamma(1e10)", lambda: np.random.standard_gamma(1e10, size=5))

subsection("standard_gamma() - Errors")
test_error_expected("standard_gamma(0)", lambda: np.random.standard_gamma(0, size=5), "ValueError")
test_error_expected("standard_gamma(-1)", lambda: np.random.standard_gamma(-1, size=5), "ValueError")
test_error_expected("standard_gamma(nan)", lambda: np.random.standard_gamma(float('nan'), size=5))

subsection("standard_gamma() - Seeded")
test_seeded("standard_gamma(2, size=10)", 42, lambda: np.random.standard_gamma(2, size=10))

# ============================================================================
#  EXPONENTIAL
# ============================================================================
section("EXPONENTIAL")

subsection("exponential() - Basic Usage")
test("exponential()", lambda: np.random.exponential())
test("exponential(1)", lambda: np.random.exponential(1))
test("exponential(2)", lambda: np.random.exponential(2))
test("exponential(0.5)", lambda: np.random.exponential(0.5))
test("exponential(1, size=5)", lambda: np.random.exponential(1, size=5))
test("exponential(1, size=(2,3))", lambda: np.random.exponential(1, size=(2,3)))

subsection("exponential() - Edge Cases")
test("exponential(EPSILON)", lambda: np.random.exponential(np.finfo(float).eps, size=5))
test("exponential(1e-10)", lambda: np.random.exponential(1e-10, size=5))
test("exponential(1e10)", lambda: np.random.exponential(1e10, size=5))

subsection("exponential() - Errors")
test_error_expected("exponential(0)", lambda: np.random.exponential(0, size=5), "ValueError")
test_error_expected("exponential(-1)", lambda: np.random.exponential(-1, size=5), "ValueError")
test_error_expected("exponential(nan)", lambda: np.random.exponential(float('nan'), size=5))
test_error_expected("exponential(inf)", lambda: np.random.exponential(float('inf'), size=5))

subsection("exponential() - Seeded")
test_seeded("exponential(1, size=10)", 42, lambda: np.random.exponential(1, size=10))

# ============================================================================
#  STANDARD_EXPONENTIAL
# ============================================================================
section("STANDARD_EXPONENTIAL")

subsection("standard_exponential() - Size Variations")
test("standard_exponential()", lambda: np.random.standard_exponential())
test("standard_exponential(None)", lambda: np.random.standard_exponential(None))
test("standard_exponential(5)", lambda: np.random.standard_exponential(5))
test("standard_exponential((2,3))", lambda: np.random.standard_exponential((2,3)))
test("standard_exponential((0,))", lambda: np.random.standard_exponential((0,)))
test_error_expected("standard_exponential(-1)", lambda: np.random.standard_exponential(-1), "ValueError")

subsection("standard_exponential() - Seeded")
test_seeded("standard_exponential(10)", 42, lambda: np.random.standard_exponential(10))

# ============================================================================
#  POISSON
# ============================================================================
section("POISSON")

subsection("poisson() - Basic Usage")
test("poisson()", lambda: np.random.poisson())
test("poisson(1)", lambda: np.random.poisson(1))
test("poisson(5)", lambda: np.random.poisson(5))
test("poisson(10)", lambda: np.random.poisson(10))
test("poisson(0.5)", lambda: np.random.poisson(0.5))
test("poisson(100)", lambda: np.random.poisson(100))
test("poisson(1, size=5)", lambda: np.random.poisson(1, size=5))
test("poisson(1, size=(2,3))", lambda: np.random.poisson(1, size=(2,3)))

subsection("poisson() - Edge Cases")
test("poisson(0)", lambda: np.random.poisson(0, size=5))  # All zeros
test("poisson(EPSILON)", lambda: np.random.poisson(np.finfo(float).eps, size=5))
test("poisson(1e-10)", lambda: np.random.poisson(1e-10, size=5))
test("poisson(1000)", lambda: np.random.poisson(1000, size=5))
test("poisson(1e10)", lambda: np.random.poisson(1e10, size=5))

subsection("poisson() - Errors")
test_error_expected("poisson(-1)", lambda: np.random.poisson(-1, size=5), "ValueError")
test_error_expected("poisson(nan)", lambda: np.random.poisson(float('nan'), size=5))
test_error_expected("poisson(inf)", lambda: np.random.poisson(float('inf'), size=5))

subsection("poisson() - Seeded")
test_seeded("poisson(5, size=10)", 42, lambda: np.random.poisson(5, size=10))

# ============================================================================
#  BINOMIAL
# ============================================================================
section("BINOMIAL")

subsection("binomial() - Basic Usage")
test("binomial(10, 0.5)", lambda: np.random.binomial(10, 0.5))
test("binomial(1, 0.5)", lambda: np.random.binomial(1, 0.5))  # Bernoulli
test("binomial(100, 0.1)", lambda: np.random.binomial(100, 0.1))
test("binomial(100, 0.9)", lambda: np.random.binomial(100, 0.9))
test("binomial(10, 0.5, size=5)", lambda: np.random.binomial(10, 0.5, size=5))
test("binomial(10, 0.5, size=(2,3))", lambda: np.random.binomial(10, 0.5, size=(2,3)))

subsection("binomial() - Edge Cases")
test("binomial(0, 0.5)", lambda: np.random.binomial(0, 0.5, size=5))  # All zeros
test("binomial(10, 0)", lambda: np.random.binomial(10, 0, size=5))  # All zeros
test("binomial(10, 1)", lambda: np.random.binomial(10, 1, size=5))  # All n
test("binomial(10, 0.0)", lambda: np.random.binomial(10, 0.0, size=5))
test("binomial(10, 1.0)", lambda: np.random.binomial(10, 1.0, size=5))
test("binomial(1000000, 0.5)", lambda: np.random.binomial(1000000, 0.5, size=5))

subsection("binomial() - Errors")
test_error_expected("binomial(-1, 0.5)", lambda: np.random.binomial(-1, 0.5, size=5), "ValueError")
test_error_expected("binomial(10, -0.1)", lambda: np.random.binomial(10, -0.1, size=5), "ValueError")
test_error_expected("binomial(10, 1.1)", lambda: np.random.binomial(10, 1.1, size=5), "ValueError")
test_error_expected("binomial(10, nan)", lambda: np.random.binomial(10, float('nan'), size=5))

subsection("binomial() - Seeded")
test_seeded("binomial(10, 0.5, size=10)", 42, lambda: np.random.binomial(10, 0.5, size=10))

# ============================================================================
#  NEGATIVE_BINOMIAL
# ============================================================================
section("NEGATIVE_BINOMIAL")

subsection("negative_binomial() - Basic Usage")
test("negative_binomial(1, 0.5)", lambda: np.random.negative_binomial(1, 0.5))
test("negative_binomial(10, 0.5)", lambda: np.random.negative_binomial(10, 0.5))
test("negative_binomial(1, 0.1)", lambda: np.random.negative_binomial(1, 0.1))
test("negative_binomial(1, 0.9)", lambda: np.random.negative_binomial(1, 0.9))
test("negative_binomial(10, 0.5, size=5)", lambda: np.random.negative_binomial(10, 0.5, size=5))
test("negative_binomial(10, 0.5, size=(2,3))", lambda: np.random.negative_binomial(10, 0.5, size=(2,3)))

subsection("negative_binomial() - Edge Cases")
test("negative_binomial(1, EPSILON)", lambda: np.random.negative_binomial(1, np.finfo(float).eps, size=5))
test("negative_binomial(1, 1-EPSILON)", lambda: np.random.negative_binomial(1, 1-np.finfo(float).eps, size=5))
test("negative_binomial(0.5, 0.5) non-int n", lambda: np.random.negative_binomial(0.5, 0.5, size=5))

subsection("negative_binomial() - Errors")
test_error_expected("negative_binomial(0, 0.5)", lambda: np.random.negative_binomial(0, 0.5, size=5), "ValueError")
test_error_expected("negative_binomial(-1, 0.5)", lambda: np.random.negative_binomial(-1, 0.5, size=5), "ValueError")
test_error_expected("negative_binomial(1, 0)", lambda: np.random.negative_binomial(1, 0, size=5), "ValueError")
test_error_expected("negative_binomial(1, 1)", lambda: np.random.negative_binomial(1, 1, size=5), "ValueError")
test_error_expected("negative_binomial(1, -0.1)", lambda: np.random.negative_binomial(1, -0.1, size=5), "ValueError")
test_error_expected("negative_binomial(1, 1.1)", lambda: np.random.negative_binomial(1, 1.1, size=5), "ValueError")

subsection("negative_binomial() - Seeded")
test_seeded("negative_binomial(10, 0.5, size=10)", 42, lambda: np.random.negative_binomial(10, 0.5, size=10))

# ============================================================================
#  GEOMETRIC
# ============================================================================
section("GEOMETRIC")

subsection("geometric() - Basic Usage")
test("geometric(0.5)", lambda: np.random.geometric(0.5))
test("geometric(0.1)", lambda: np.random.geometric(0.1))
test("geometric(0.9)", lambda: np.random.geometric(0.9))
test("geometric(0.5, size=5)", lambda: np.random.geometric(0.5, size=5))
test("geometric(0.5, size=(2,3))", lambda: np.random.geometric(0.5, size=(2,3)))

subsection("geometric() - Edge Cases")
test("geometric(1)", lambda: np.random.geometric(1, size=5))  # Always 1
test("geometric(EPSILON)", lambda: np.random.geometric(np.finfo(float).eps, size=5))
test("geometric(1-EPSILON)", lambda: np.random.geometric(1-np.finfo(float).eps, size=5))

subsection("geometric() - Errors")
test_error_expected("geometric(0)", lambda: np.random.geometric(0, size=5), "ValueError")
test_error_expected("geometric(-0.1)", lambda: np.random.geometric(-0.1, size=5), "ValueError")
test_error_expected("geometric(1.1)", lambda: np.random.geometric(1.1, size=5), "ValueError")
test_error_expected("geometric(nan)", lambda: np.random.geometric(float('nan'), size=5))

subsection("geometric() - Seeded")
test_seeded("geometric(0.5, size=10)", 42, lambda: np.random.geometric(0.5, size=10))

# ============================================================================
#  HYPERGEOMETRIC
# ============================================================================
section("HYPERGEOMETRIC")

subsection("hypergeometric() - Basic Usage")
test("hypergeometric(10, 5, 3)", lambda: np.random.hypergeometric(10, 5, 3))
test("hypergeometric(100, 50, 25)", lambda: np.random.hypergeometric(100, 50, 25))
test("hypergeometric(10, 5, 3, size=5)", lambda: np.random.hypergeometric(10, 5, 3, size=5))
test("hypergeometric(10, 5, 3, size=(2,3))", lambda: np.random.hypergeometric(10, 5, 3, size=(2,3)))

subsection("hypergeometric() - Edge Cases")
test("hypergeometric(0, 5, 0)", lambda: np.random.hypergeometric(0, 5, 0, size=5))
test("hypergeometric(10, 0, 0)", lambda: np.random.hypergeometric(10, 0, 0, size=5))
test("hypergeometric(10, 5, 0)", lambda: np.random.hypergeometric(10, 5, 0, size=5))  # nsample=0
test("hypergeometric(10, 5, 15)", lambda: np.random.hypergeometric(10, 5, 15, size=5))  # nsample=ngood+nbad

subsection("hypergeometric() - Errors")
test_error_expected("hypergeometric(-1, 5, 3)", lambda: np.random.hypergeometric(-1, 5, 3, size=5), "ValueError")
test_error_expected("hypergeometric(10, -1, 3)", lambda: np.random.hypergeometric(10, -1, 3, size=5), "ValueError")
test_error_expected("hypergeometric(10, 5, -1)", lambda: np.random.hypergeometric(10, 5, -1, size=5), "ValueError")
test_error_expected("hypergeometric(10, 5, 20) nsample>ngood+nbad", lambda: np.random.hypergeometric(10, 5, 20, size=5), "ValueError")

subsection("hypergeometric() - Seeded")
test_seeded("hypergeometric(10, 5, 3, size=10)", 42, lambda: np.random.hypergeometric(10, 5, 3, size=10))

# ============================================================================
#  CHISQUARE
# ============================================================================
section("CHISQUARE")

subsection("chisquare() - Basic Usage")
test("chisquare(1)", lambda: np.random.chisquare(1))
test("chisquare(2)", lambda: np.random.chisquare(2))
test("chisquare(10)", lambda: np.random.chisquare(10))
test("chisquare(0.5)", lambda: np.random.chisquare(0.5))
test("chisquare(1, size=5)", lambda: np.random.chisquare(1, size=5))
test("chisquare(1, size=(2,3))", lambda: np.random.chisquare(1, size=(2,3)))

subsection("chisquare() - Edge Cases")
test("chisquare(EPSILON)", lambda: np.random.chisquare(np.finfo(float).eps, size=5))
test("chisquare(1e-10)", lambda: np.random.chisquare(1e-10, size=5))
test("chisquare(1e10)", lambda: np.random.chisquare(1e10, size=5))

subsection("chisquare() - Errors")
test_error_expected("chisquare(0)", lambda: np.random.chisquare(0, size=5), "ValueError")
test_error_expected("chisquare(-1)", lambda: np.random.chisquare(-1, size=5), "ValueError")
test_error_expected("chisquare(nan)", lambda: np.random.chisquare(float('nan'), size=5))

subsection("chisquare() - Seeded")
test_seeded("chisquare(5, size=10)", 42, lambda: np.random.chisquare(5, size=10))

# ============================================================================
#  NONCENTRAL_CHISQUARE
# ============================================================================
section("NONCENTRAL_CHISQUARE")

subsection("noncentral_chisquare() - Basic Usage")
test("noncentral_chisquare(1, 1)", lambda: np.random.noncentral_chisquare(1, 1))
test("noncentral_chisquare(5, 2)", lambda: np.random.noncentral_chisquare(5, 2))
test("noncentral_chisquare(1, 1, size=5)", lambda: np.random.noncentral_chisquare(1, 1, size=5))
test("noncentral_chisquare(1, 1, size=(2,3))", lambda: np.random.noncentral_chisquare(1, 1, size=(2,3)))

subsection("noncentral_chisquare() - Edge Cases")
test("noncentral_chisquare(1, 0)", lambda: np.random.noncentral_chisquare(1, 0, size=5))  # Reduces to chisquare
test("noncentral_chisquare(EPSILON, 1)", lambda: np.random.noncentral_chisquare(np.finfo(float).eps, 1, size=5))
test("noncentral_chisquare(1, EPSILON)", lambda: np.random.noncentral_chisquare(1, np.finfo(float).eps, size=5))

subsection("noncentral_chisquare() - Errors")
test_error_expected("noncentral_chisquare(0, 1)", lambda: np.random.noncentral_chisquare(0, 1, size=5), "ValueError")
test_error_expected("noncentral_chisquare(-1, 1)", lambda: np.random.noncentral_chisquare(-1, 1, size=5), "ValueError")
test_error_expected("noncentral_chisquare(1, -1)", lambda: np.random.noncentral_chisquare(1, -1, size=5), "ValueError")

subsection("noncentral_chisquare() - Seeded")
test_seeded("noncentral_chisquare(5, 2, size=10)", 42, lambda: np.random.noncentral_chisquare(5, 2, size=10))

# ============================================================================
#  F
# ============================================================================
section("F (Fisher)")

subsection("f() - Basic Usage")
test("f(1, 1)", lambda: np.random.f(1, 1))
test("f(5, 10)", lambda: np.random.f(5, 10))
test("f(10, 5)", lambda: np.random.f(10, 5))
test("f(1, 1, size=5)", lambda: np.random.f(1, 1, size=5))
test("f(1, 1, size=(2,3))", lambda: np.random.f(1, 1, size=(2,3)))

subsection("f() - Edge Cases")
test("f(EPSILON, 1)", lambda: np.random.f(np.finfo(float).eps, 1, size=5))
test("f(1, EPSILON)", lambda: np.random.f(1, np.finfo(float).eps, size=5))
test("f(1e-10, 1)", lambda: np.random.f(1e-10, 1, size=5))
test("f(1e10, 1e10)", lambda: np.random.f(1e10, 1e10, size=5))

subsection("f() - Errors")
test_error_expected("f(0, 1)", lambda: np.random.f(0, 1, size=5), "ValueError")
test_error_expected("f(1, 0)", lambda: np.random.f(1, 0, size=5), "ValueError")
test_error_expected("f(-1, 1)", lambda: np.random.f(-1, 1, size=5), "ValueError")
test_error_expected("f(1, -1)", lambda: np.random.f(1, -1, size=5), "ValueError")

subsection("f() - Seeded")
test_seeded("f(5, 10, size=10)", 42, lambda: np.random.f(5, 10, size=10))

# ============================================================================
#  NONCENTRAL_F
# ============================================================================
section("NONCENTRAL_F")

subsection("noncentral_f() - Basic Usage")
test("noncentral_f(1, 1, 1)", lambda: np.random.noncentral_f(1, 1, 1))
test("noncentral_f(5, 10, 2)", lambda: np.random.noncentral_f(5, 10, 2))
test("noncentral_f(1, 1, 1, size=5)", lambda: np.random.noncentral_f(1, 1, 1, size=5))
test("noncentral_f(1, 1, 1, size=(2,3))", lambda: np.random.noncentral_f(1, 1, 1, size=(2,3)))

subsection("noncentral_f() - Edge Cases")
test("noncentral_f(1, 1, 0)", lambda: np.random.noncentral_f(1, 1, 0, size=5))  # Reduces to F
test("noncentral_f(1, 1, EPSILON)", lambda: np.random.noncentral_f(1, 1, np.finfo(float).eps, size=5))

subsection("noncentral_f() - Errors")
test_error_expected("noncentral_f(0, 1, 1)", lambda: np.random.noncentral_f(0, 1, 1, size=5), "ValueError")
test_error_expected("noncentral_f(1, 0, 1)", lambda: np.random.noncentral_f(1, 0, 1, size=5), "ValueError")
test_error_expected("noncentral_f(1, 1, -1)", lambda: np.random.noncentral_f(1, 1, -1, size=5), "ValueError")

subsection("noncentral_f() - Seeded")
test_seeded("noncentral_f(5, 10, 2, size=10)", 42, lambda: np.random.noncentral_f(5, 10, 2, size=10))

# ============================================================================
#  STANDARD_T (Student's t)
# ============================================================================
section("STANDARD_T")

subsection("standard_t() - Basic Usage")
test("standard_t(1)", lambda: np.random.standard_t(1))  # Cauchy
test("standard_t(2)", lambda: np.random.standard_t(2))
test("standard_t(10)", lambda: np.random.standard_t(10))
test("standard_t(100)", lambda: np.random.standard_t(100))  # Approaches normal
test("standard_t(1, size=5)", lambda: np.random.standard_t(1, size=5))
test("standard_t(1, size=(2,3))", lambda: np.random.standard_t(1, size=(2,3)))

subsection("standard_t() - Edge Cases")
test("standard_t(EPSILON)", lambda: np.random.standard_t(np.finfo(float).eps, size=5))
test("standard_t(0.5)", lambda: np.random.standard_t(0.5, size=5))
test("standard_t(1e10)", lambda: np.random.standard_t(1e10, size=5))

subsection("standard_t() - Errors")
test_error_expected("standard_t(0)", lambda: np.random.standard_t(0, size=5), "ValueError")
test_error_expected("standard_t(-1)", lambda: np.random.standard_t(-1, size=5), "ValueError")
test_error_expected("standard_t(nan)", lambda: np.random.standard_t(float('nan'), size=5))

subsection("standard_t() - Seeded")
test_seeded("standard_t(5, size=10)", 42, lambda: np.random.standard_t(5, size=10))

# ============================================================================
#  STANDARD_CAUCHY
# ============================================================================
section("STANDARD_CAUCHY")

subsection("standard_cauchy() - Size Variations")
test("standard_cauchy()", lambda: np.random.standard_cauchy())
test("standard_cauchy(None)", lambda: np.random.standard_cauchy(None))
test("standard_cauchy(5)", lambda: np.random.standard_cauchy(5))
test("standard_cauchy((2,3))", lambda: np.random.standard_cauchy((2,3)))
test("standard_cauchy((0,))", lambda: np.random.standard_cauchy((0,)))
test_error_expected("standard_cauchy(-1)", lambda: np.random.standard_cauchy(-1), "ValueError")

subsection("standard_cauchy() - Seeded")
test_seeded("standard_cauchy(10)", 42, lambda: np.random.standard_cauchy(10))

# ============================================================================
#  LAPLACE
# ============================================================================
section("LAPLACE")

subsection("laplace() - Basic Usage")
test("laplace()", lambda: np.random.laplace())
test("laplace(0, 1)", lambda: np.random.laplace(0, 1))
test("laplace(5, 2)", lambda: np.random.laplace(5, 2))
test("laplace(-5, 0.5)", lambda: np.random.laplace(-5, 0.5))
test("laplace(0, 1, size=5)", lambda: np.random.laplace(0, 1, size=5))
test("laplace(0, 1, size=(2,3))", lambda: np.random.laplace(0, 1, size=(2,3)))

subsection("laplace() - Edge Cases")
test("laplace(0, EPSILON)", lambda: np.random.laplace(0, np.finfo(float).eps, size=5))
test("laplace(0, 1e-10)", lambda: np.random.laplace(0, 1e-10, size=5))
test("laplace(1e308, 1)", lambda: np.random.laplace(1e308, 1, size=5))
test("laplace(0, 1e308)", lambda: np.random.laplace(0, 1e308, size=5))

subsection("laplace() - Errors")
test_error_expected("laplace(0, 0)", lambda: np.random.laplace(0, 0, size=5), "ValueError")
test_error_expected("laplace(0, -1)", lambda: np.random.laplace(0, -1, size=5), "ValueError")
test_error_expected("laplace(nan, 1)", lambda: np.random.laplace(float('nan'), 1, size=5))
test_error_expected("laplace(0, nan)", lambda: np.random.laplace(0, float('nan'), size=5))

subsection("laplace() - Seeded")
test_seeded("laplace(0, 1, size=10)", 42, lambda: np.random.laplace(0, 1, size=10))

# ============================================================================
#  LOGISTIC
# ============================================================================
section("LOGISTIC")

subsection("logistic() - Basic Usage")
test("logistic()", lambda: np.random.logistic())
test("logistic(0, 1)", lambda: np.random.logistic(0, 1))
test("logistic(5, 2)", lambda: np.random.logistic(5, 2))
test("logistic(0, 1, size=5)", lambda: np.random.logistic(0, 1, size=5))
test("logistic(0, 1, size=(2,3))", lambda: np.random.logistic(0, 1, size=(2,3)))

subsection("logistic() - Edge Cases")
test("logistic(0, EPSILON)", lambda: np.random.logistic(0, np.finfo(float).eps, size=5))
test("logistic(0, 1e-10)", lambda: np.random.logistic(0, 1e-10, size=5))
test("logistic(1e308, 1)", lambda: np.random.logistic(1e308, 1, size=5))

subsection("logistic() - Errors")
test_error_expected("logistic(0, 0)", lambda: np.random.logistic(0, 0, size=5), "ValueError")
test_error_expected("logistic(0, -1)", lambda: np.random.logistic(0, -1, size=5), "ValueError")

subsection("logistic() - Seeded")
test_seeded("logistic(0, 1, size=10)", 42, lambda: np.random.logistic(0, 1, size=10))

# ============================================================================
#  GUMBEL
# ============================================================================
section("GUMBEL")

subsection("gumbel() - Basic Usage")
test("gumbel()", lambda: np.random.gumbel())
test("gumbel(0, 1)", lambda: np.random.gumbel(0, 1))
test("gumbel(5, 2)", lambda: np.random.gumbel(5, 2))
test("gumbel(0, 1, size=5)", lambda: np.random.gumbel(0, 1, size=5))
test("gumbel(0, 1, size=(2,3))", lambda: np.random.gumbel(0, 1, size=(2,3)))

subsection("gumbel() - Edge Cases")
test("gumbel(0, EPSILON)", lambda: np.random.gumbel(0, np.finfo(float).eps, size=5))
test("gumbel(1e308, 1)", lambda: np.random.gumbel(1e308, 1, size=5))

subsection("gumbel() - Errors")
test_error_expected("gumbel(0, 0)", lambda: np.random.gumbel(0, 0, size=5), "ValueError")
test_error_expected("gumbel(0, -1)", lambda: np.random.gumbel(0, -1, size=5), "ValueError")

subsection("gumbel() - Seeded")
test_seeded("gumbel(0, 1, size=10)", 42, lambda: np.random.gumbel(0, 1, size=10))

# ============================================================================
#  LOGNORMAL
# ============================================================================
section("LOGNORMAL")

subsection("lognormal() - Basic Usage")
test("lognormal()", lambda: np.random.lognormal())
test("lognormal(0, 1)", lambda: np.random.lognormal(0, 1))
test("lognormal(5, 2)", lambda: np.random.lognormal(5, 2))
test("lognormal(-5, 0.5)", lambda: np.random.lognormal(-5, 0.5))
test("lognormal(0, 1, size=5)", lambda: np.random.lognormal(0, 1, size=5))
test("lognormal(0, 1, size=(2,3))", lambda: np.random.lognormal(0, 1, size=(2,3)))

subsection("lognormal() - Edge Cases")
test("lognormal(0, EPSILON)", lambda: np.random.lognormal(0, np.finfo(float).eps, size=5))
test("lognormal(0, 1e-10)", lambda: np.random.lognormal(0, 1e-10, size=5))
test("lognormal(700, 1)", lambda: np.random.lognormal(700, 1, size=5))  # Near overflow

subsection("lognormal() - Errors")
test_error_expected("lognormal(0, 0)", lambda: np.random.lognormal(0, 0, size=5), "ValueError")
test_error_expected("lognormal(0, -1)", lambda: np.random.lognormal(0, -1, size=5), "ValueError")

subsection("lognormal() - Seeded")
test_seeded("lognormal(0, 1, size=10)", 42, lambda: np.random.lognormal(0, 1, size=10))

# ============================================================================
#  LOGSERIES
# ============================================================================
section("LOGSERIES")

subsection("logseries() - Basic Usage")
test("logseries(0.5)", lambda: np.random.logseries(0.5))
test("logseries(0.1)", lambda: np.random.logseries(0.1))
test("logseries(0.9)", lambda: np.random.logseries(0.9))
test("logseries(0.5, size=5)", lambda: np.random.logseries(0.5, size=5))
test("logseries(0.5, size=(2,3))", lambda: np.random.logseries(0.5, size=(2,3)))

subsection("logseries() - Edge Cases")
test("logseries(EPSILON)", lambda: np.random.logseries(np.finfo(float).eps, size=5))
test("logseries(1-EPSILON)", lambda: np.random.logseries(1-np.finfo(float).eps, size=5))
test("logseries(1e-10)", lambda: np.random.logseries(1e-10, size=5))
test("logseries(0.9999999)", lambda: np.random.logseries(0.9999999, size=5))

subsection("logseries() - Errors")
test_error_expected("logseries(0)", lambda: np.random.logseries(0, size=5), "ValueError")
test_error_expected("logseries(1)", lambda: np.random.logseries(1, size=5), "ValueError")
test_error_expected("logseries(-0.1)", lambda: np.random.logseries(-0.1, size=5), "ValueError")
test_error_expected("logseries(1.1)", lambda: np.random.logseries(1.1, size=5), "ValueError")

subsection("logseries() - Seeded")
test_seeded("logseries(0.5, size=10)", 42, lambda: np.random.logseries(0.5, size=10))

# ============================================================================
#  PARETO
# ============================================================================
section("PARETO")

subsection("pareto() - Basic Usage")
test("pareto(1)", lambda: np.random.pareto(1))
test("pareto(2)", lambda: np.random.pareto(2))
test("pareto(5)", lambda: np.random.pareto(5))
test("pareto(0.5)", lambda: np.random.pareto(0.5))
test("pareto(1, size=5)", lambda: np.random.pareto(1, size=5))
test("pareto(1, size=(2,3))", lambda: np.random.pareto(1, size=(2,3)))

subsection("pareto() - Edge Cases")
test("pareto(EPSILON)", lambda: np.random.pareto(np.finfo(float).eps, size=5))
test("pareto(1e-10)", lambda: np.random.pareto(1e-10, size=5))
test("pareto(1e10)", lambda: np.random.pareto(1e10, size=5))

subsection("pareto() - Errors")
test_error_expected("pareto(0)", lambda: np.random.pareto(0, size=5), "ValueError")
test_error_expected("pareto(-1)", lambda: np.random.pareto(-1, size=5), "ValueError")

subsection("pareto() - Seeded")
test_seeded("pareto(2, size=10)", 42, lambda: np.random.pareto(2, size=10))

# ============================================================================
#  POWER
# ============================================================================
section("POWER")

subsection("power() - Basic Usage")
test("power(1)", lambda: np.random.power(1))  # Uniform on [0, 1]
test("power(2)", lambda: np.random.power(2))
test("power(5)", lambda: np.random.power(5))
test("power(0.5)", lambda: np.random.power(0.5))
test("power(1, size=5)", lambda: np.random.power(1, size=5))
test("power(1, size=(2,3))", lambda: np.random.power(1, size=(2,3)))

subsection("power() - Edge Cases")
test("power(EPSILON)", lambda: np.random.power(np.finfo(float).eps, size=5))
test("power(1e-10)", lambda: np.random.power(1e-10, size=5))
test("power(1e10)", lambda: np.random.power(1e10, size=5))

subsection("power() - Errors")
test_error_expected("power(0)", lambda: np.random.power(0, size=5), "ValueError")
test_error_expected("power(-1)", lambda: np.random.power(-1, size=5), "ValueError")

subsection("power() - Seeded")
test_seeded("power(2, size=10)", 42, lambda: np.random.power(2, size=10))

# ============================================================================
#  RAYLEIGH
# ============================================================================
section("RAYLEIGH")

subsection("rayleigh() - Basic Usage")
test("rayleigh()", lambda: np.random.rayleigh())
test("rayleigh(1)", lambda: np.random.rayleigh(1))
test("rayleigh(2)", lambda: np.random.rayleigh(2))
test("rayleigh(0.5)", lambda: np.random.rayleigh(0.5))
test("rayleigh(1, size=5)", lambda: np.random.rayleigh(1, size=5))
test("rayleigh(1, size=(2,3))", lambda: np.random.rayleigh(1, size=(2,3)))

subsection("rayleigh() - Edge Cases")
test("rayleigh(EPSILON)", lambda: np.random.rayleigh(np.finfo(float).eps, size=5))
test("rayleigh(1e-10)", lambda: np.random.rayleigh(1e-10, size=5))
test("rayleigh(1e10)", lambda: np.random.rayleigh(1e10, size=5))

subsection("rayleigh() - Errors")
test_error_expected("rayleigh(0)", lambda: np.random.rayleigh(0, size=5), "ValueError")
test_error_expected("rayleigh(-1)", lambda: np.random.rayleigh(-1, size=5), "ValueError")

subsection("rayleigh() - Seeded")
test_seeded("rayleigh(1, size=10)", 42, lambda: np.random.rayleigh(1, size=10))

# ============================================================================
#  TRIANGULAR
# ============================================================================
section("TRIANGULAR")

subsection("triangular() - Basic Usage")
test("triangular(0, 0.5, 1)", lambda: np.random.triangular(0, 0.5, 1))
test("triangular(-1, 0, 1)", lambda: np.random.triangular(-1, 0, 1))
test("triangular(0, 1, 1) mode==right", lambda: np.random.triangular(0, 1, 1, size=5))
test("triangular(0, 0, 1) mode==left", lambda: np.random.triangular(0, 0, 1, size=5))
test("triangular(0, 0.5, 1, size=5)", lambda: np.random.triangular(0, 0.5, 1, size=5))
test("triangular(0, 0.5, 1, size=(2,3))", lambda: np.random.triangular(0, 0.5, 1, size=(2,3)))

subsection("triangular() - Edge Cases")
test("triangular(0, 0, 0) degenerate", lambda: np.random.triangular(0, 0, 0, size=5))  # All zeros
test("triangular(5, 5, 5) degenerate", lambda: np.random.triangular(5, 5, 5, size=5))  # All fives
test("triangular(-1e308, 0, 1e308)", lambda: np.random.triangular(-1e308, 0, 1e308, size=5))

subsection("triangular() - Errors")
test_error_expected("triangular(1, 0, 2) mode<left", lambda: np.random.triangular(1, 0, 2, size=5), "ValueError")
test_error_expected("triangular(0, 3, 2) mode>right", lambda: np.random.triangular(0, 3, 2, size=5), "ValueError")
test_error_expected("triangular(2, 1, 0) left>right", lambda: np.random.triangular(2, 1, 0, size=5), "ValueError")

subsection("triangular() - Seeded")
test_seeded("triangular(0, 0.5, 1, size=10)", 42, lambda: np.random.triangular(0, 0.5, 1, size=10))

# ============================================================================
#  VONMISES
# ============================================================================
section("VONMISES")

subsection("vonmises() - Basic Usage")
test("vonmises(0, 1)", lambda: np.random.vonmises(0, 1))
test("vonmises(np.pi, 1)", lambda: np.random.vonmises(np.pi, 1))
test("vonmises(0, 0.5)", lambda: np.random.vonmises(0, 0.5))
test("vonmises(0, 4)", lambda: np.random.vonmises(0, 4))
test("vonmises(0, 1, size=5)", lambda: np.random.vonmises(0, 1, size=5))
test("vonmises(0, 1, size=(2,3))", lambda: np.random.vonmises(0, 1, size=(2,3)))

subsection("vonmises() - Edge Cases")
test("vonmises(0, 0)", lambda: np.random.vonmises(0, 0, size=5))  # Uniform on circle
test("vonmises(0, EPSILON)", lambda: np.random.vonmises(0, np.finfo(float).eps, size=5))
test("vonmises(0, 1e10)", lambda: np.random.vonmises(0, 1e10, size=5))  # Very concentrated
test("vonmises(2*np.pi, 1)", lambda: np.random.vonmises(2*np.pi, 1, size=5))  # mu outside [-pi, pi]
test("vonmises(-2*np.pi, 1)", lambda: np.random.vonmises(-2*np.pi, 1, size=5))

subsection("vonmises() - Errors")
test_error_expected("vonmises(0, -1)", lambda: np.random.vonmises(0, -1, size=5), "ValueError")

subsection("vonmises() - Seeded")
test_seeded("vonmises(0, 1, size=10)", 42, lambda: np.random.vonmises(0, 1, size=10))

# ============================================================================
#  WALD (Inverse Gaussian)
# ============================================================================
section("WALD")

subsection("wald() - Basic Usage")
test("wald(1, 1)", lambda: np.random.wald(1, 1))
test("wald(2, 1)", lambda: np.random.wald(2, 1))
test("wald(1, 2)", lambda: np.random.wald(1, 2))
test("wald(0.5, 0.5)", lambda: np.random.wald(0.5, 0.5))
test("wald(1, 1, size=5)", lambda: np.random.wald(1, 1, size=5))
test("wald(1, 1, size=(2,3))", lambda: np.random.wald(1, 1, size=(2,3)))

subsection("wald() - Edge Cases")
test("wald(EPSILON, 1)", lambda: np.random.wald(np.finfo(float).eps, 1, size=5))
test("wald(1, EPSILON)", lambda: np.random.wald(1, np.finfo(float).eps, size=5))
test("wald(1e10, 1)", lambda: np.random.wald(1e10, 1, size=5))
test("wald(1, 1e10)", lambda: np.random.wald(1, 1e10, size=5))

subsection("wald() - Errors")
test_error_expected("wald(0, 1)", lambda: np.random.wald(0, 1, size=5), "ValueError")
test_error_expected("wald(1, 0)", lambda: np.random.wald(1, 0, size=5), "ValueError")
test_error_expected("wald(-1, 1)", lambda: np.random.wald(-1, 1, size=5), "ValueError")
test_error_expected("wald(1, -1)", lambda: np.random.wald(1, -1, size=5), "ValueError")

subsection("wald() - Seeded")
test_seeded("wald(1, 1, size=10)", 42, lambda: np.random.wald(1, 1, size=10))

# ============================================================================
#  WEIBULL
# ============================================================================
section("WEIBULL")

subsection("weibull() - Basic Usage")
test("weibull(1)", lambda: np.random.weibull(1))  # Exponential
test("weibull(2)", lambda: np.random.weibull(2))  # Rayleigh-like
test("weibull(5)", lambda: np.random.weibull(5))
test("weibull(0.5)", lambda: np.random.weibull(0.5))
test("weibull(1, size=5)", lambda: np.random.weibull(1, size=5))
test("weibull(1, size=(2,3))", lambda: np.random.weibull(1, size=(2,3)))

subsection("weibull() - Edge Cases")
test("weibull(EPSILON)", lambda: np.random.weibull(np.finfo(float).eps, size=5))
test("weibull(1e-10)", lambda: np.random.weibull(1e-10, size=5))
test("weibull(1e10)", lambda: np.random.weibull(1e10, size=5))

subsection("weibull() - Errors")
test_error_expected("weibull(0)", lambda: np.random.weibull(0, size=5), "ValueError")
test_error_expected("weibull(-1)", lambda: np.random.weibull(-1, size=5), "ValueError")

subsection("weibull() - Seeded")
test_seeded("weibull(2, size=10)", 42, lambda: np.random.weibull(2, size=10))

# ============================================================================
#  ZIPF
# ============================================================================
section("ZIPF")

subsection("zipf() - Basic Usage")
test("zipf(2)", lambda: np.random.zipf(2))
test("zipf(1.5)", lambda: np.random.zipf(1.5))
test("zipf(3)", lambda: np.random.zipf(3))
test("zipf(2, size=5)", lambda: np.random.zipf(2, size=5))
test("zipf(2, size=(2,3))", lambda: np.random.zipf(2, size=(2,3)))

subsection("zipf() - Edge Cases")
test("zipf(1+EPSILON)", lambda: np.random.zipf(1+np.finfo(float).eps, size=5))
test("zipf(1.0001)", lambda: np.random.zipf(1.0001, size=5))
test("zipf(1e10)", lambda: np.random.zipf(1e10, size=5))

subsection("zipf() - Errors")
test_error_expected("zipf(1)", lambda: np.random.zipf(1, size=5), "ValueError")  # Must be > 1
test_error_expected("zipf(0.5)", lambda: np.random.zipf(0.5, size=5), "ValueError")
test_error_expected("zipf(0)", lambda: np.random.zipf(0, size=5), "ValueError")
test_error_expected("zipf(-1)", lambda: np.random.zipf(-1, size=5), "ValueError")

subsection("zipf() - Seeded")
test_seeded("zipf(2, size=10)", 42, lambda: np.random.zipf(2, size=10))

# ============================================================================
#  CHOICE
# ============================================================================
section("CHOICE")

subsection("choice() - From Integer")
test("choice(10)", lambda: np.random.choice(10))
test("choice(10, size=5)", lambda: np.random.choice(10, size=5))
test("choice(10, size=(2,3))", lambda: np.random.choice(10, size=(2,3)))
test("choice(10, replace=True)", lambda: np.random.choice(10, size=5, replace=True))
test("choice(10, replace=False)", lambda: np.random.choice(10, size=5, replace=False))
test("choice(5, size=5, replace=False)", lambda: np.random.choice(5, size=5, replace=False))  # Exact fit
test("choice(1)", lambda: np.random.choice(1))  # Single element
test("choice(1, size=5)", lambda: np.random.choice(1, size=5))  # All zeros

subsection("choice() - From Array")
test("choice([1,2,3,4,5])", lambda: np.random.choice([1,2,3,4,5]))
test("choice(np.arange(10))", lambda: np.random.choice(np.arange(10)))
test("choice(['a','b','c'])", lambda: np.random.choice(['a','b','c']))
test("choice([1,2,3], size=5)", lambda: np.random.choice([1,2,3], size=5))
test("choice([1,2,3], replace=False)", lambda: np.random.choice([1,2,3], size=3, replace=False))

subsection("choice() - With Probabilities")
test("choice(5, p=[0.1,0.2,0.3,0.3,0.1])", lambda: np.random.choice(5, size=10, p=[0.1,0.2,0.3,0.3,0.1]))
test("choice([1,2,3], p=[0.5,0.3,0.2])", lambda: np.random.choice([1,2,3], size=10, p=[0.5,0.3,0.2]))
test("choice(3, p=[1,0,0])", lambda: np.random.choice(3, size=5, p=[1,0,0]))  # Deterministic
test("choice(3, p=[0,0,1])", lambda: np.random.choice(3, size=5, p=[0,0,1]))  # Deterministic

subsection("choice() - Edge Cases")
test("choice(10, size=0)", lambda: np.random.choice(10, size=0))
test("choice(10, size=(0,))", lambda: np.random.choice(10, size=(0,)))
test("choice(10, size=(2,0))", lambda: np.random.choice(10, size=(2,0)))

subsection("choice() - Errors")
test_error_expected("choice(0)", lambda: np.random.choice(0), "ValueError")
test_error_expected("choice(-1)", lambda: np.random.choice(-1), "ValueError")
test_error_expected("choice([])", lambda: np.random.choice([]), "ValueError")
test_error_expected("choice(5, size=10, replace=False)", lambda: np.random.choice(5, size=10, replace=False), "ValueError")
test_error_expected("choice(5, p=[0.1,0.2,0.3])", lambda: np.random.choice(5, p=[0.1,0.2,0.3]), "ValueError")  # Wrong length
test_error_expected("choice(3, p=[0.5,0.5,0.5])", lambda: np.random.choice(3, p=[0.5,0.5,0.5]), "ValueError")  # Sum != 1
test_error_expected("choice(3, p=[-0.1,0.6,0.5])", lambda: np.random.choice(3, p=[-0.1,0.6,0.5]), "ValueError")  # Negative

subsection("choice() - Seeded")
test_seeded("choice(100, size=10)", 42, lambda: np.random.choice(100, size=10))
test_seeded("choice([1,2,3,4,5], size=10)", 42, lambda: np.random.choice([1,2,3,4,5], size=10))

# ============================================================================
#  SHUFFLE
# ============================================================================
section("SHUFFLE")

subsection("shuffle() - 1D Arrays")
def test_shuffle_1d():
    arr = np.arange(10)
    np.random.seed(42)
    np.random.shuffle(arr)
    return arr
test("shuffle(arange(10))", test_shuffle_1d)

def test_shuffle_1d_copy():
    arr = np.arange(10).copy()
    original = arr.copy()
    np.random.shuffle(arr)
    return arr, "differs from original:", not np.array_equal(arr, original)
test("shuffle modifies in-place", test_shuffle_1d_copy)

subsection("shuffle() - 2D Arrays (shuffle along axis 0)")
def test_shuffle_2d():
    arr = np.arange(12).reshape(4, 3)
    np.random.seed(42)
    np.random.shuffle(arr)
    return arr
test("shuffle(4x3 array) shuffles rows", test_shuffle_2d)

def test_shuffle_2d_cols_unchanged():
    arr = np.arange(12).reshape(4, 3)
    np.random.seed(42)
    np.random.shuffle(arr)
    # Each row should still be consecutive (just reordered)
    for row in arr:
        if not (row[1] == row[0] + 1 and row[2] == row[1] + 1):
            return False, arr
    return True, arr
test("shuffle(2D) preserves row contents", test_shuffle_2d_cols_unchanged)

subsection("shuffle() - Edge Cases")
def test_shuffle_single():
    arr = np.array([42])
    np.random.shuffle(arr)
    return arr
test("shuffle([42]) single element", test_shuffle_single)

def test_shuffle_empty():
    arr = np.array([])
    np.random.shuffle(arr)
    return arr
test("shuffle([]) empty array", test_shuffle_empty)

def test_shuffle_2d_single_row():
    arr = np.array([[1, 2, 3]])
    np.random.shuffle(arr)
    return arr
test("shuffle([[1,2,3]]) single row", test_shuffle_2d_single_row)

subsection("shuffle() - Errors")
test_error_expected("shuffle(scalar)", lambda: np.random.shuffle(np.array(5)), "ValueError")

subsection("shuffle() - Seeded Reproducibility")
def test_shuffle_seeded():
    arr1 = np.arange(10)
    np.random.seed(42)
    np.random.shuffle(arr1)

    arr2 = np.arange(10)
    np.random.seed(42)
    np.random.shuffle(arr2)
    return np.array_equal(arr1, arr2), arr1, arr2
test("shuffle seeded reproducibility", test_shuffle_seeded)

# ============================================================================
#  PERMUTATION
# ============================================================================
section("PERMUTATION")

subsection("permutation() - From Integer")
test("permutation(10)", lambda: np.random.permutation(10))
test("permutation(1)", lambda: np.random.permutation(1))
test("permutation(0)", lambda: np.random.permutation(0))

subsection("permutation() - From Array")
test("permutation([1,2,3,4,5])", lambda: np.random.permutation([1,2,3,4,5]))
test("permutation(np.arange(10))", lambda: np.random.permutation(np.arange(10)))

subsection("permutation() - Returns Copy (doesn't modify original)")
def test_permutation_copy():
    original = np.arange(10)
    result = np.random.permutation(original)
    return np.array_equal(original, np.arange(10)), original, result
test("permutation returns copy, original unchanged", test_permutation_copy)

subsection("permutation() - 2D Arrays")
def test_permutation_2d():
    arr = np.arange(12).reshape(4, 3)
    np.random.seed(42)
    result = np.random.permutation(arr)
    return result
test("permutation(4x3) permutes rows", test_permutation_2d)

subsection("permutation() - Seeded")
test_seeded("permutation(10)", 42, lambda: np.random.permutation(10))
test_seeded("permutation([1,2,3,4,5])", 42, lambda: np.random.permutation([1,2,3,4,5]))

# ============================================================================
#  DIRICHLET
# ============================================================================
section("DIRICHLET")

subsection("dirichlet() - Basic Usage")
test("dirichlet([1,1,1])", lambda: np.random.dirichlet([1,1,1]))
test("dirichlet([0.5,0.5])", lambda: np.random.dirichlet([0.5,0.5]))
test("dirichlet([1,2,3,4])", lambda: np.random.dirichlet([1,2,3,4]))
test("dirichlet([10,10,10])", lambda: np.random.dirichlet([10,10,10]))
test("dirichlet([1,1,1], size=5)", lambda: np.random.dirichlet([1,1,1], size=5))
test("dirichlet([1,1,1], size=(2,3))", lambda: np.random.dirichlet([1,1,1], size=(2,3)))

subsection("dirichlet() - Output Sum Check")
def test_dirichlet_sum():
    samples = np.random.dirichlet([1,2,3], size=10)
    sums = samples.sum(axis=-1)
    return np.allclose(sums, 1.0), sums
test("dirichlet samples sum to 1", test_dirichlet_sum)

subsection("dirichlet() - Edge Cases")
test("dirichlet([EPSILON,EPSILON])", lambda: np.random.dirichlet([np.finfo(float).eps, np.finfo(float).eps], size=5))
test("dirichlet([1e-10,1e-10])", lambda: np.random.dirichlet([1e-10, 1e-10], size=5))
test("dirichlet([1e10,1e10])", lambda: np.random.dirichlet([1e10, 1e10], size=5))
test("dirichlet([1])", lambda: np.random.dirichlet([1], size=5))  # Single alpha

subsection("dirichlet() - Errors")
test_error_expected("dirichlet([])", lambda: np.random.dirichlet([]), "ValueError")
test_error_expected("dirichlet([0,1])", lambda: np.random.dirichlet([0,1], size=5), "ValueError")
test_error_expected("dirichlet([-1,1])", lambda: np.random.dirichlet([-1,1], size=5), "ValueError")
test_error_expected("dirichlet([1,nan])", lambda: np.random.dirichlet([1,float('nan')], size=5))

subsection("dirichlet() - Seeded")
test_seeded("dirichlet([1,2,3], size=5)", 42, lambda: np.random.dirichlet([1,2,3], size=5))

# ============================================================================
#  MULTINOMIAL
# ============================================================================
section("MULTINOMIAL")

subsection("multinomial() - Basic Usage")
test("multinomial(10, [0.2,0.3,0.5])", lambda: np.random.multinomial(10, [0.2,0.3,0.5]))
test("multinomial(100, [0.5,0.5])", lambda: np.random.multinomial(100, [0.5,0.5]))
test("multinomial(10, [1/3,1/3,1/3])", lambda: np.random.multinomial(10, [1/3,1/3,1/3]))
test("multinomial(10, [0.2,0.3,0.5], size=5)", lambda: np.random.multinomial(10, [0.2,0.3,0.5], size=5))
test("multinomial(10, [0.2,0.3,0.5], size=(2,3))", lambda: np.random.multinomial(10, [0.2,0.3,0.5], size=(2,3)))

subsection("multinomial() - Output Sum Check")
def test_multinomial_sum():
    samples = np.random.multinomial(100, [0.2,0.3,0.5], size=10)
    sums = samples.sum(axis=-1)
    return np.all(sums == 100), sums
test("multinomial samples sum to n", test_multinomial_sum)

subsection("multinomial() - Edge Cases")
test("multinomial(0, [0.5,0.5])", lambda: np.random.multinomial(0, [0.5,0.5], size=5))  # All zeros
test("multinomial(10, [1,0,0])", lambda: np.random.multinomial(10, [1,0,0], size=5))  # Deterministic
test("multinomial(10, [0,0,1])", lambda: np.random.multinomial(10, [0,0,1], size=5))  # Deterministic
test("multinomial(1, [0.5,0.5])", lambda: np.random.multinomial(1, [0.5,0.5], size=10))  # n=1

subsection("multinomial() - Errors")
test_error_expected("multinomial(-1, [0.5,0.5])", lambda: np.random.multinomial(-1, [0.5,0.5], size=5), "ValueError")
test_error_expected("multinomial(10, [])", lambda: np.random.multinomial(10, []), "ValueError")
test_error_expected("multinomial(10, [0.5,0.6])", lambda: np.random.multinomial(10, [0.5,0.6], size=5), "ValueError")  # Sum > 1
test_error_expected("multinomial(10, [-0.1,0.6,0.5])", lambda: np.random.multinomial(10, [-0.1,0.6,0.5], size=5), "ValueError")

subsection("multinomial() - Seeded")
test_seeded("multinomial(10, [0.2,0.3,0.5], size=5)", 42, lambda: np.random.multinomial(10, [0.2,0.3,0.5], size=5))

# ============================================================================
#  MULTIVARIATE_NORMAL
# ============================================================================
section("MULTIVARIATE_NORMAL")

subsection("multivariate_normal() - Basic Usage")
test("multivariate_normal([0,0], [[1,0],[0,1]])", lambda: np.random.multivariate_normal([0,0], [[1,0],[0,1]]))
test("multivariate_normal([1,2], [[1,0.5],[0.5,1]])", lambda: np.random.multivariate_normal([1,2], [[1,0.5],[0.5,1]]))
test("multivariate_normal([0,0,0], np.eye(3))", lambda: np.random.multivariate_normal([0,0,0], np.eye(3)))
test("multivariate_normal([0,0], [[1,0],[0,1]], size=5)", lambda: np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=5))
test("multivariate_normal([0,0], [[1,0],[0,1]], size=(2,3))", lambda: np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=(2,3)))

subsection("multivariate_normal() - Edge Cases")
test("multivariate_normal 1D", lambda: np.random.multivariate_normal([0], [[1]], size=5))
test("multivariate_normal near-singular cov", lambda: np.random.multivariate_normal([0,0], [[1,0.9999],[0.9999,1]], size=5))
test("multivariate_normal diagonal cov", lambda: np.random.multivariate_normal([0,0], [[2,0],[0,3]], size=5))
test("multivariate_normal zero mean", lambda: np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=5))

subsection("multivariate_normal() - Errors")
test_error_expected("multivariate_normal mean/cov mismatch", lambda: np.random.multivariate_normal([0,0,0], [[1,0],[0,1]]), "ValueError")
test_error_expected("multivariate_normal non-square cov", lambda: np.random.multivariate_normal([0,0], [[1,0,0],[0,1,0]]), "ValueError")
test_error_expected("multivariate_normal non-symmetric cov", lambda: np.random.multivariate_normal([0,0], [[1,0.5],[0.3,1]]))  # May or may not error

subsection("multivariate_normal() - Seeded")
test_seeded("multivariate_normal([0,0], [[1,0],[0,1]], size=5)", 42, lambda: np.random.multivariate_normal([0,0], [[1,0],[0,1]], size=5))

# ============================================================================
#  SPECIAL: BERNOULLI (not in standard NumPy, but in NumSharp)
# ============================================================================
section("BERNOULLI (NumSharp-specific)")

print("Note: bernoulli() is NumSharp-specific, equivalent to binomial(1, p)")
print("Testing binomial(1, p) as proxy:")

subsection("binomial(1, p) as Bernoulli")
test("binomial(1, 0.5, size=10) - Bernoulli", lambda: np.random.binomial(1, 0.5, size=10))
test("binomial(1, 0.1, size=10) - Bernoulli", lambda: np.random.binomial(1, 0.1, size=10))
test("binomial(1, 0.9, size=10) - Bernoulli", lambda: np.random.binomial(1, 0.9, size=10))
test("binomial(1, 0, size=10) - Bernoulli all 0", lambda: np.random.binomial(1, 0, size=10))
test("binomial(1, 1, size=10) - Bernoulli all 1", lambda: np.random.binomial(1, 1, size=10))

# ============================================================================
#  SIZE PARAMETER VARIATIONS (Cross-cutting)
# ============================================================================
section("SIZE PARAMETER VARIATIONS")

subsection("Size=None returns scalar")
test("uniform() returns scalar", lambda: type(np.random.uniform()).__name__)
test("normal() returns scalar", lambda: type(np.random.normal()).__name__)
test("randn() returns scalar", lambda: type(np.random.randn()).__name__)
test("randint(10) returns scalar", lambda: type(np.random.randint(10)).__name__)

subsection("Size=() returns 0-d array")
test("uniform(size=()) 0-d array", lambda: np.random.uniform(size=()))
test("normal(size=()) 0-d array", lambda: np.random.normal(size=()))
test("randint(10, size=()) 0-d array", lambda: np.random.randint(10, size=()))

def show_0d_properties():
    arr = np.random.uniform(size=())
    return f"shape={arr.shape}, ndim={arr.ndim}, size={arr.size}, item={arr.item()}"
test("0-d array properties", show_0d_properties)

subsection("Size=0 returns empty array")
test("uniform(size=0)", lambda: np.random.uniform(size=0))
test("uniform(size=(0,))", lambda: np.random.uniform(size=(0,)))
test("uniform(size=(5,0))", lambda: np.random.uniform(size=(5,0)))
test("uniform(size=(0,5))", lambda: np.random.uniform(size=(0,5)))
test("uniform(size=(0,0))", lambda: np.random.uniform(size=(0,0)))

subsection("Size as various types")
test("uniform(size=5) int", lambda: np.random.uniform(size=5))
test("uniform(size=(5,)) tuple", lambda: np.random.uniform(size=(5,)))
test("uniform(size=[5]) list", lambda: np.random.uniform(size=[5]))
test("uniform(size=np.array([5])) array", lambda: np.random.uniform(size=np.array([5])))
test("uniform(size=np.int32(5)) np.int32", lambda: np.random.uniform(size=np.int32(5)))
test("uniform(size=np.int64(5)) np.int64", lambda: np.random.uniform(size=np.int64(5)))

subsection("Negative size errors")
test_error_expected("uniform(size=-1)", lambda: np.random.uniform(size=-1), "ValueError")
test_error_expected("uniform(size=(-1,))", lambda: np.random.uniform(size=(-1,)), "ValueError")
test_error_expected("uniform(size=(5,-1))", lambda: np.random.uniform(size=(5,-1)), "ValueError")

# ============================================================================
#  DTYPE OUTPUT VERIFICATION
# ============================================================================
section("DTYPE OUTPUT VERIFICATION")

subsection("Default dtypes")
test("rand() dtype", lambda: np.random.rand(5).dtype)
test("randn() dtype", lambda: np.random.randn(5).dtype)
test("uniform() dtype", lambda: np.random.uniform(size=5).dtype)
test("normal() dtype", lambda: np.random.normal(size=5).dtype)
test("randint() dtype", lambda: np.random.randint(10, size=5).dtype)
test("choice() dtype from int", lambda: np.random.choice(10, size=5).dtype)
test("binomial() dtype", lambda: np.random.binomial(10, 0.5, size=5).dtype)
test("poisson() dtype", lambda: np.random.poisson(5, size=5).dtype)

subsection("randint explicit dtypes")
for dtype in [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]:
    test(f"randint dtype={dtype.__name__}", lambda d=dtype: np.random.randint(10, size=5, dtype=d).dtype)

# ============================================================================
#  SEQUENCE REPRODUCIBILITY (Critical for NumSharp matching)
# ============================================================================
section("SEQUENCE REPRODUCIBILITY")

subsection("Multiple calls with same seed produce same sequence")
def test_sequence_reproducibility():
    np.random.seed(42)
    seq1 = [np.random.random() for _ in range(10)]
    np.random.seed(42)
    seq2 = [np.random.random() for _ in range(10)]
    return seq1 == seq2, seq1
test("Sequential random() calls", test_sequence_reproducibility)

def test_mixed_sequence():
    np.random.seed(42)
    a = np.random.random()
    b = np.random.randint(100)
    c = np.random.randn()
    d = np.random.uniform(0, 10)
    np.random.seed(42)
    a2 = np.random.random()
    b2 = np.random.randint(100)
    c2 = np.random.randn()
    d2 = np.random.uniform(0, 10)
    return (a==a2, b==b2, c==c2, d==d2), (a, b, c, d)
test("Mixed call sequence", test_mixed_sequence)

subsection("Exact values for reference (seed=42)")
test_seeded("5 random() values", 42, lambda: [np.random.random() for _ in range(5)])
test_seeded("5 randint(100) values", 42, lambda: [np.random.randint(100) for _ in range(5)])
test_seeded("5 randn() values", 42, lambda: [np.random.randn() for _ in range(5)])
test_seeded("uniform(0,100,5) values", 42, lambda: np.random.uniform(0, 100, 5))
test_seeded("normal(0,1,5) values", 42, lambda: np.random.normal(0, 1, 5))

# ============================================================================
#  GAUSSIAN CACHING (NumPy uses polar method with caching)
# ============================================================================
section("GAUSSIAN CACHING")

subsection("State includes Gaussian cache")
def test_gauss_cache():
    np.random.seed(42)
    # Generate one Gaussian (consumes 2 uniforms, caches second Gaussian)
    g1 = np.random.randn()
    state = np.random.get_state()
    print(f"     After 1 randn: has_gauss={state[3]}, cached={state[4]:.6f}")

    # Generate second Gaussian (uses cached value)
    g2 = np.random.randn()
    state = np.random.get_state()
    print(f"     After 2 randn: has_gauss={state[3]}, cached={state[4]:.6f}")

    return g1, g2
test("Gaussian cache state", test_gauss_cache)

# ============================================================================
#  FINAL SUMMARY
# ============================================================================
section("BATTLETEST COMPLETE")
print("This battletest covers:")
print("- 40+ distribution functions")
print("- Parameter validation (bounds, types, edge cases)")
print("- Size parameter variations (None, int, tuple, 0, negative)")
print("- dtype verification")
print("- Seed reproducibility")
print("- State save/restore")
print("- Gaussian caching behavior")
print("- Error messages and exception types")
print()
print("Use this output to verify NumSharp implementation matches NumPy exactly.")
