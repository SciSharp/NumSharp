using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using TorchSharp;
using static TorchSharp.torch;

namespace NumSharp.Tests.Interop.TorchSharp
{
    /// <summary>
    ///     Shared checks of the state-matrix suites: write-through probes in both directions, layout comparisons that skip
    ///     what cannot matter (the stride of a length-1 axis, any stride of an empty array), and a failure collector so one
    ///     state reports every dtype and mode that failed at once.
    /// </summary>
    public static class StateMatrixSupport
    {
        /// <summary>The three codec modes, in the order the suites run them.</summary>
        public static readonly NumSharp.Interop.TorchSharp.TorchCodecMode[] Modes =
        {
            NumSharp.Interop.TorchSharp.TorchCodecMode.Auto,
            NumSharp.Interop.TorchSharp.TorchCodecMode.View,
            NumSharp.Interop.TorchSharp.TorchCodecMode.Copy,
        };

        /// <summary>
        ///     Whether two element-stride vectors address the same elements of a shape: equal on every axis longer than 1, and
        ///     trivially for an empty shape (nothing is addressed).
        /// </summary>
        /// <param name="shape">The dimensions.</param>
        /// <param name="a">One stride vector.</param>
        /// <param name="b">The other.</param>
        /// <returns>Whether they agree where it matters.</returns>
        public static bool SameStrides(long[] shape, long[] a, long[] b)
        {
            if (shape.Any(d => d == 0))
                return true;
            if (a.Length != shape.Length || b.Length != shape.Length)
                return false;
            for (int i = 0; i < shape.Length; i++)
                if (shape[i] > 1 && a[i] != b[i])
                    return false;
            return true;
        }

        /// <summary>Whether any axis longer than 1 has stride 0 (a broadcast / expanded layout, which NumSharp marks read-only).</summary>
        /// <param name="shape">The dimensions.</param>
        /// <param name="strides">The element strides.</param>
        /// <returns>Whether a stride-0 axis repeats elements.</returns>
        public static bool HasRepeatedAxis(long[] shape, long[] strides)
        {
            for (int i = 0; i < shape.Length; i++)
                if (shape[i] > 1 && strides[i] == 0)
                    return true;
            return false;
        }

        /// <summary>
        ///     The real part of a tensor's first logical element as a double, read through a chain of 0-index selects inside its
        ///     own DisposeScope.
        /// </summary>
        /// <param name="t">The tensor (at least one element).</param>
        /// <returns>The value.</returns>
        public static double FirstValue(Tensor t)
        {
            using var scope = torch.NewDisposeScope();
            Tensor e = t;
            for (long i = 0; i < t.ndim; i++)
                e = e[0];
            if (e.is_complex())
                e = e.real;
            return e.to_type(ScalarType.Float64).item<double>();
        }

        /// <summary>A value that differs from <paramref name="current"/> in every dtype (7 and 0, so a bool flips too).</summary>
        /// <param name="current">The element's current value.</param>
        /// <returns>0 when the element is non-zero, else 7.</returns>
        public static double Other(double current) => current == 0 ? 7 : 0;

        /// <summary>
        ///     Writes <paramref name="value"/> into a tensor's first logical element through a chain of 0-index selects (a view
        ///     of the very element, whatever the strides), inside its own DisposeScope.
        /// </summary>
        /// <param name="t">The tensor (at least one element).</param>
        /// <param name="value">The value (7 lands as true in a bool tensor).</param>
        public static void TorchWriteFirst(Tensor t, double value)
        {
            using var scope = torch.NewDisposeScope();
            Tensor e = t;
            for (long i = 0; i < t.ndim; i++)
                e = e[0];
            e.fill_(value);
        }

        /// <summary>
        ///     Writes <paramref name="value"/> into a NumSharp array's first logical element through a slice view (a full integer
        ///     index would read a scalar COPY, NumPy's rule), converted to the array's dtype by NumPy's assignment.
        /// </summary>
        /// <param name="nd">The array (at least one element, writeable).</param>
        /// <param name="value">The value (0 lands as false in a bool array).</param>
        public static void NumSharpWriteFirst(NDArray nd, double value)
        {
            using NDArray v = np.full(new Shape(), value, nd.dtype);
            if (nd.ndim == 0)
            {
                np.copyto(nd, v);
                return;
            }
            using NDArray first = nd[string.Join(",", Enumerable.Repeat("0:1", nd.ndim))];
            np.copyto(first, v);
        }

        /// <summary>Collects the failures of one state across its dtypes and modes, for one assertion at the end.</summary>
        public sealed class Failures
        {
            /// <summary>The messages so far.</summary>
            private readonly List<string> _messages = new List<string>();

            /// <summary>Records a failure unless the condition holds.</summary>
            /// <param name="condition">What must hold.</param>
            /// <param name="message">What to report when it does not.</param>
            public void Check(bool condition, string message)
            {
                if (!condition)
                    _messages.Add(message);
            }

            /// <summary>Records an unexpected exception.</summary>
            /// <param name="where">The case.</param>
            /// <param name="ex">The exception.</param>
            public void Unexpected(string where, Exception ex) => _messages.Add($"{where}: unexpected {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");

            /// <summary>Whether nothing failed.</summary>
            public bool None => _messages.Count == 0;

            /// <summary>Every failure, one per line.</summary>
            /// <returns>The report.</returns>
            public override string ToString() => $"{_messages.Count} failure(s):\n" + string.Join("\n", _messages);
        }
    }
}
