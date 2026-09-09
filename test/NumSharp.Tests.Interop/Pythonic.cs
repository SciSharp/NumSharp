using System;
using System.Runtime.InteropServices;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;
using Python.Runtime;

// =========================================================================================
//  Test-side pythonic numpy — the interop's Pythonic.cs style, extended to what the codec
//  byte-contract tests need: build REAL numpy arrays in clean C#, read their dtype / shape /
//  strides / bytes, and compare both sides BYTE-FOR-BYTE.
//
//  The numpy module lives at `Python.np` (deliberately beside `Python.Runtime`) so the two
//  sides read for what they are — `Python.np.array(...)` is Python's numpy, while `np.*` stays
//  NumSharp:
//
//      using (Gil())
//      {
//          using PyObject a = Python.np.eval("np.arange(12, dtype='<i4').reshape(3,4).T");  // numpy
//          using NDArray nd = a.ToNDArray();                                                // crossed
//          ByteContract.AssertSameBytes(nd, a);                                             // byte-for-byte
//      }
//
//  GIL: every member here calls the C-API, so it must run under the GIL — exactly like the
//  interop's Pythonic.cs. The high-level ByteContract.Assert* helpers acquire it themselves.
// =========================================================================================

namespace Python
{
    /// <summary><c>import numpy as np</c>, the test flavour: clean builders returning real numpy arrays.
    /// Sits in the <c>Python</c> namespace (beside <c>Python.Runtime</c>) so test code reads
    /// <c>Python.np.array(...)</c>. Call under the GIL.</summary>
    internal static class np
    {
        /// <summary>The <c>numpy</c> module (CPython caches it; the wrapper is short-lived — dispose after use).</summary>
        private static PyObject Module => Runtime.Py.Import("numpy");

        /// <summary>
        ///     Evaluate a numpy expression with <c>np</c> bound — the escape hatch for the layouts C#
        ///     cannot spell (slices, transposes, broadcasts). The result outlives the throwaway scope.
        /// </summary>
        internal static PyObject eval(string npExpr)
        {
            using var scope = Runtime.Py.CreateScope();
            scope.Exec("import numpy as np");
            return scope.Eval(npExpr);
        }

        /// <summary>
        ///     Evaluate a numpy expression with named operands pre-bound — the multi-operand form of
        ///     <see cref="eval"/>. Lets a test hand REAL numpy the arrays this interop exported and let
        ///     numpy's own kernels compute over them: <c>with("np.matmul(a, b)", ("a", va), ("b", vb))</c>.
        ///     The bound operands stay owned by the caller (the throwaway scope only increfs them); the
        ///     result outlives the scope.
        /// </summary>
        internal static PyObject with(string npExpr, params (string name, PyObject val)[] vars)
        {
            using var scope = Runtime.Py.CreateScope();
            scope.Exec("import numpy as np");
            foreach (var (name, val) in vars)
                scope.Set(name, val);
            return scope.Eval(npExpr);
        }

        /// <summary>True iff the numpy expression (evaluated with <c>np</c> bound) is truthy — for
        /// semantic probes like <c>truthy("np.isnan(a).all()", ("a", exported))</c>.</summary>
        internal static bool truthy(string npExpr, params (string name, PyObject val)[] vars)
        {
            using PyObject r = with($"bool({npExpr})", vars);
            return r.As<bool>();
        }

        /// <summary>
        ///     Run setup STATEMENTS (with <c>np</c> bound) then return <paramref name="resultVar"/> — for
        ///     arrays a single expression cannot build (e.g. one flipped to <c>writeable = False</c>).
        /// </summary>
        internal static PyObject evalStmts(string setup, string resultVar)
        {
            using var scope = Runtime.Py.CreateScope();
            scope.Exec("import numpy as np");
            scope.Exec(setup);
            return scope.Eval(resultVar);
        }

        internal static PyObject arange(long n, string dtype = "int64") => eval($"np.arange({n}).astype('{dtype}')");
        internal static PyObject array(string literal, string dtype) => eval($"np.array({literal}, dtype='{dtype}')");
        internal static PyObject zeros(string shape, string dtype) => eval($"np.zeros({shape}, dtype='{dtype}')");
        internal static PyObject scalar(string dtype, string value) => eval($"np.{dtype}({value})");

        internal static PyObject asarray(PyObject o) { using var m = Module; return m.InvokeMethod("asarray", o); }

        internal static bool array_equal(PyObject a, PyObject b)
        { using var m = Module; using var r = m.InvokeMethod("array_equal", a, b); return r.As<bool>(); }

        internal static bool shares_memory(PyObject a, PyObject b)
        { using var m = Module; using var r = m.InvokeMethod("shares_memory", a, b); return r.As<bool>(); }
    }

    /// <summary><c>import torch</c>, the test flavour — reach a tensor's members as the Python reads
    /// (<c>Python.torch.Tensor.data_ptr_t(t)</c>). Sits beside <see cref="np"/> in the <c>Python</c>
    /// namespace so torch tests read for what they drive. Call under the GIL.</summary>
    internal static class torch
    {
        /// <summary><c>str(torch.__version__)</c> — the installed torch version string.</summary>
        internal static string version()
        {
            using var scope = Runtime.Py.CreateScope();
            scope.Exec("import torch");
            using PyObject v = scope.Eval("str(torch.__version__)");
            return v.As<string>();
        }

        /// <summary>The <c>torch.Tensor</c> instance members the tests reach, spelled unbound as Python
        /// spells them (<c>torch.Tensor.data_ptr(t)</c>). Each returned <see cref="PyObject"/> is owned
        /// by the caller (<c>using</c>).</summary>
        internal static class Tensor
        {
            /// <summary><c>tensor.dtype</c> — the tensor's dtype object.</summary>
            internal static PyObject dtype(PyObject t) => t.GetAttr("dtype");

            /// <summary><c>tensor.data_ptr()</c> — the tensor's data address as a CLR long.</summary>
            internal static long data_ptr_t(PyObject t)
            { using PyObject p = t.InvokeMethod("data_ptr"); return p.As<long>(); }

            /// <summary><c>tensor.resize_(n)</c> — in-place resize (raises when the storage is not resizable).</summary>
            internal static PyObject resize_(PyObject t, long n)
            { using PyObject size = n.ToPython(); return t.InvokeMethod("resize_", size); }
        }
    }

    /// <summary><c>import pandas as pd</c>, the test flavour. Sits beside <see cref="np"/> in the
    /// <c>Python</c> namespace. Call under the GIL.</summary>
    internal static class pandas
    {
        /// <summary><c>str(pd.__version__)</c> — the installed pandas version string.</summary>
        internal static string version()
        {
            using var scope = Runtime.Py.CreateScope();
            scope.Exec("import pandas as pd");
            using PyObject v = scope.Eval("str(pd.__version__)");
            return v.As<string>();
        }

        /// <summary><c>pd.Series(data)</c> — a Series over the given array. The result outlives the
        /// throwaway module handle; the caller owns it (<c>using</c>).</summary>
        internal static PyObject Series(PyObject data)
        {
            using PyObject mod = Runtime.Py.Import("pandas");
            using PyObject ctor = mod.GetAttr("Series");
            return ctor.Invoke(data);
        }
    }
}

namespace NumSharp.Tests.Interop
{
    /// <summary>The array-instance side: <c>arr.dtype.str</c>, <c>arr.shape</c>, <c>arr.astype(...)</c>,
    /// <c>arr.tobytes('C')</c>, spelled as clean C# extension calls. Call under the GIL.</summary>
    internal static class NumpyExtensions
    {
        internal static string dtype_str(this PyObject a)
        { using var d = a.GetAttr("dtype"); using var s = d.GetAttr("str"); return s.As<string>(); }

        internal static long[] shape_dims(this PyObject a)
        { using var s = a.GetAttr("shape"); return TupleToLongs(s); }

        internal static long[] strides_bytes(this PyObject a)
        { using var s = a.GetAttr("strides"); return TupleToLongs(s); }

        internal static bool writeable(this PyObject a)
        { using var f = a.GetAttr("flags"); using var w = f.GetAttr("writeable"); return w.As<bool>(); }

        internal static PyObject astype(this PyObject a, string dtype) => a.InvokeMethod("astype", dtype.ToPython());

        /// <summary>Canonical C-order bytes (<c>np.ascontiguousarray(a).tobytes('C')</c>) — the byte oracle.</summary>
        internal static byte[] bytes_c(this PyObject a)
        {
            using var np = Py.Import("numpy");
            using PyObject cont = np.InvokeMethod("ascontiguousarray", a);
            using PyObject b = cont.InvokeMethod("tobytes", "C".ToPython());
            return b.As<byte[]>();
        }

        private static long[] TupleToLongs(PyObject tupleLike)
        {
            using var tup = PyTuple.AsTuple(tupleLike);
            int n = (int)tup.Length();
            var values = new long[n];
            for (int i = 0; i < n; i++) { using var e = tup[i]; values[i] = e.As<long>(); }
            return values;
        }
    }

    /// <summary>
    ///     Byte-for-byte equivalence between the numpy side and the NumSharp side — the strongest
    ///     equality: it catches dtype width, endianness, layout linearization and padding that a value
    ///     compare (<c>array_equal</c>) silently accepts.
    /// </summary>
    internal static class ByteContract
    {
        /// <summary>
        ///     A NumSharp array's canonical C-order bytes, read INDEPENDENTLY of the interop's
        ///     <c>ToNumpy</c> (through NumSharp's own <c>ascontiguousarray</c> + a raw buffer copy), so a
        ///     byte match is a genuine cross-check of the crossing, not a tautology.
        /// </summary>
        internal static unsafe byte[] NsBytes(NDArray nd)
        {
            NDArray c = NumSharp.np.ascontiguousarray(nd);
            long n = c.size * c.dtypesize;
            var bytes = new byte[n];
            if (n > 0)
            {
                byte* p = (byte*)c.Storage.Address + c.Shape.Offset * c.dtypesize;
                Marshal.Copy((nint)p, bytes, 0, checked((int)n));
            }

            return bytes;
        }

        /// <summary>Assert the NumSharp array is byte-identical to the numpy array it should equal. Call under the GIL.</summary>
        internal static void AssertSameBytes(NDArray ns, PyObject expected, string because = "")
            => NsBytes(ns).Should().Equal(expected.bytes_c(), because);

        /// <summary>Assert two numpy arrays are byte-identical (encode: exported vs the expected form). Call under the GIL.</summary>
        internal static void AssertSameBytes(PyObject a, PyObject b, string because = "")
            => a.bytes_c().Should().Equal(b.bytes_c(), because);
    }
}
