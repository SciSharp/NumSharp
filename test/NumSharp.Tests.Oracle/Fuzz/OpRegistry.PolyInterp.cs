using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp;

namespace NumSharp.Tests.Fuzz
{
    /// <summary>
    ///     <c>numpy.polynomial.chebyshev.chebinterpolate</c> (plan U9): the C# twins of <c>gen_oracle.py</c>'s
    ///     <c>_PI_FUNCS</c> menu. A corpus case names its callable (<c>params.func</c>); each twin computes, from the
    ///     Chebyshev points, exactly the values NumPy's Python function does — the same operations in the same operand order —
    ///     so the replay compares chebinterpolate itself, not the menu.
    /// </summary>
    public static partial class OpRegistry
    {
        /// <summary>
        ///     The menu, by name: <c>func(x, args)</c> twins of the Python callables (null for NumPy's <c>None</c> func).
        ///     Results keep the Python function's KIND — an ndarray, a Python list (<c>object[]</c>), a tuple
        ///     (<see cref="ValueTuple"/>), a Python scalar (a boxed C# double / long / Complex) — since np.dot's conversion
        ///     of each kind is part of the contract.
        /// </summary>
        private static readonly Dictionary<string, Func<NDArray, object[], object>> ChebInterpolateFuncs = new(StringComparer.Ordinal)
        {
            ["tanh_half"] = (x, _) => np.tanh(x) + 0.5,
            ["sin"] = (x, _) => np.sin(x),
            ["exp"] = (x, _) => np.exp(x),
            // x * x * x - 2.0 * x + 1.0: Python's left-to-right evaluation, the scalar operands in their positions.
            ["cubic"] = (x, _) => x * x * x - 2.0 * x + 1.0,
            ["abs"] = (x, _) => np.abs(x),
            // np.exp(1j * x): the complex scalar is the FIRST operand of the multiply, as in Python.
            ["expi"] = (x, _) => np.exp(np.multiply(NDArray.Scalar(new Complex(0, 1)), x)),
            ["stack2"] = (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1),
            ["stack2f"] = (x, _) => np.asfortranarray(np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1)),
            // np.sin(x)[None, :, None] * np.ones((2, 1, 3)): the (1, n, 1) view times ones, broadcast to (2, n, 3).
            ["cube3d"] = (x, _) => np.multiply(np.sin(x).reshape(1, x.size, 1), np.ones(new Shape(2, 1, 3))),
            // [float(v) for v in np.sin(x)]: a Python list of Python floats.
            ["list"] = (x, _) => Array.ConvertAll(np.sin(x).ToArray<double>(), v => (object)v),
            // tuple(float(v) for v in np.sin(x)): a Python tuple of Python floats.
            ["tuple"] = (x, _) =>
            {
                object[] items = Array.ConvertAll(np.sin(x).ToArray<double>(), v => (object)v);
                return PolySequence.MakeTuple(items, items.Length);
            },
            ["pyfloat"] = (x, _) => 2.0,
            ["pyint"] = (x, _) => 2L,
            ["pycomplex"] = (x, _) => new Complex(1, 2),
            ["negzero"] = (x, _) => np.full(new Shape(x.size), -0.0),
            ["zerod"] = (x, _) => np.array(3.0),
            ["f32"] = (x, _) => np.sin(x).astype(np.float32),
            ["f16"] = (x, _) => np.sin(x).astype(np.float16),
            ["i64"] = (x, _) => (x * 10).astype(np.int64),
            ["i8"] = (x, _) => (x * 10).astype(np.int8),
            ["u64"] = (x, _) => (np.abs(x) * 10).astype(np.uint64),
            ["u16"] = (x, _) => (np.abs(x) * 10).astype(np.uint16),
            ["bool"] = (x, _) => x > 0,
            ["strided"] = (x, _) => np.repeat(np.sin(x), 2)["::2"],
            ["reversed"] = (x, _) => np.sin(x)["::-1"].copy()["::-1"],
            ["x"] = (x, _) => x,
            ["wronglen"] = (x, _) => np.ones(new Shape(x.size + 1)),
            ["wrong2d"] = (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 0),
            // [[1.0]] + [[1.0, 2.0]] * (len(x) - 1): a ragged Python list (from two points on).
            ["ragged"] = (x, _) =>
            {
                var rows = new object[x.size];
                rows[0] = new object[] { 1.0 };
                for (int i = 1; i < rows.Length; i++)
                    rows[i] = new object[] { 1.0, 2.0 };
                return rows;
            },
            // np.sin(a * x) + b: the argument first in the multiply, the sine first in the add.
            ["sin_ab"] = (x, a) => np.add(np.sin(np.multiply(InterpArg(a[0]), x)), InterpArg(a[1])),
            ["scale_a"] = (x, a) => np.multiply(InterpArg(a[0]), x),
            ["none"] = null,
        };

        /// <summary>
        ///     One unpacked <c>args</c> item as the operand Python's arithmetic sees: an NDArray as it is (an ndarray's row),
        ///     any C# number as the 0-d array of its discovered dtype — for the menu's float64 points the result dtype and values
        ///     are the same as Python's weak scalar arithmetic.
        /// </summary>
        /// <param name="o">The item (an NDArray, or a boxed C# number).</param>
        /// <returns>The operand.</returns>
        /// <exception cref="NotSupportedException">An item that is not a number or an array.</exception>
        private static NDArray InterpArg(object o) => o as NDArray ?? PolyNumber.FromObject(o).ToNDArray();

        /// <summary>The menu entry of a case (<c>params.func</c>).</summary>
        /// <param name="name">The menu name.</param>
        /// <returns>The twin (null for <c>"none"</c>, NumPy's None func).</returns>
        /// <exception cref="NotSupportedException">A name the menu does not hold.</exception>
        private static Func<NDArray, object[], object> ChebInterpolateFunc(string name)
            => ChebInterpolateFuncs.TryGetValue(name, out var f)
                ? f
                : throw new NotSupportedException($"chebinterpolate menu function '{name}' is not registered in OpRegistry");
    }
}
