using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    // =====================================================================================
    //  Python-shaped access to the session-cached interop callables.
    //
    //  Every conversion in this package executes a short, fixed Python program (build a
    //  ctypes window, frombuffer it, stride it, root a finalizer, ...). These internal
    //  static classes let the C# call sites READ like that Python — module classes named
    //  after the modules, nested statics for submodules, and extension members on
    //  PyObject for the instance-side calls:
    //
    //      ctBuf = ctypes.c_char.mul(tailBytes).from_address(dataPtr);   # (c_char * n).from_address(p)
    //      flat  = np.frombuffer(ctBuf, source.typecode);                # np.frombuffer(buf, '<f8')
    //      arr   = np.lib.stride_tricks.as_strided(flat, dims, strides); # verbatim
    //      arr.setflags(write: false);                                   # arr.setflags(write=False)
    //      if (mv.c_contiguous) ...                                      # mv.c_contiguous
    //      using var b = mv.tobytes("C");                                # mv.tobytes('C')
    //
    //  This is a veneer, not a second cache: every member resolves through
    //  <see cref="PythonInteropRuntime"/>'s per-session cache (pre-resolved callables, interned
    //  attribute-name PyStrings, cached literals) and calls Invoke directly — no dynamic,
    //  no per-call attribute walks, no extra dispatch layers beyond an inlineable static.
    //
    //  GIL: exactly like the cache itself, every member must be called under the GIL.
    //  Scope: everything here is internal — the lowercase Python-style names (np, ctypes,
    //  weakref, builtins) can never collide with NumSharp.np or user code outside this
    //  assembly, and inside it the interop never touches NumSharp's own np class.
    // =====================================================================================

    /// <summary>
    ///     <c>import numpy as np</c> — the numpy calls the interop makes, as they read in Python.
    ///     Session-cached callables via <see cref="PythonInteropRuntime"/>; call under the GIL.
    /// </summary>
    internal static class np
    {
        /// <summary><c>np.empty(shape, dtype)</c> — dtype routed through the cached per-dtype PyString.</summary>
        internal static PyObject empty(PyObject shape, NPTypeCode dtype)
            => PythonInteropRuntime.NpEmpty.Invoke(shape, PythonInteropRuntime.DtypeString(dtype));

        /// <summary><c>np.frombuffer(buffer, dtype)</c> — dtype routed through the cached per-dtype PyString.</summary>
        internal static PyObject frombuffer(PyObject buffer, NPTypeCode dtype)
            => PythonInteropRuntime.NpFrombuffer.Invoke(buffer, PythonInteropRuntime.DtypeString(dtype));

        /// <summary><c>np.array(obj)</c> — copies by default, exactly the property the copy path relies on.</summary>
        internal static PyObject array(PyObject obj)
            => PythonInteropRuntime.NpArray.Invoke(obj);

        /// <summary><c>np.lib</c> submodule namespace.</summary>
        internal static class lib
        {
            /// <summary><c>np.lib.stride_tricks</c> submodule namespace.</summary>
            internal static class stride_tricks
            {
                /// <summary><c>np.lib.stride_tricks.as_strided(x, shape, strides)</c> — no bounds checking, by design.</summary>
                internal static PyObject as_strided(PyObject x, PyObject shape, PyObject strides)
                    => PythonInteropRuntime.NpAsStrided.Invoke(x, shape, strides);
            }
        }
    }

    /// <summary>
    ///     <c>import ctypes</c> — the raw-pointer window builder. Call under the GIL.
    /// </summary>
    internal static class ctypes
    {
        /// <summary><c>ctypes.c_char</c> — the byte scalar type the buffer windows are built from.</summary>
        internal static class c_char
        {
            /// <summary>
            ///     <c>(ctypes.c_char * n)</c> — the sized array type, via the session-cached bound
            ///     <c>__mul__</c> (the ctypes metaclass exposes the repeat operator there). The
            ///     returned handle is ONE-SHOT: <see cref="SizedArray.from_address"/> consumes it,
            ///     so the fluent chain <c>ctypes.c_char.mul(n).from_address(addr)</c> is leak-free.
            /// </summary>
            internal static SizedArray mul(long n)
            {
                using var count = new PyInt(n);
                return new SizedArray(PythonInteropRuntime.CCharMul.Invoke(count));
            }
        }

        /// <summary>
        ///     A sized ctypes array type (<c>c_char_Array_n</c>) pending its one
        ///     <see cref="from_address"/> call. Produced only by <see cref="c_char.mul"/>.
        /// </summary>
        internal readonly struct SizedArray
        {
            private readonly PyObject _type;

            internal SizedArray(PyObject type) => _type = type;

            /// <summary>
            ///     <c>array_type.from_address(address)</c> — wraps existing memory WITHOUT copying or
            ///     claiming ownership. Consumes this handle (the type wrapper is disposed either way;
            ///     the returned buffer object keeps the type alive through its own Python reference).
            /// </summary>
            internal PyObject from_address(long address)
            {
                try
                {
                    using PyObject fromAddress = _type.GetAttr(PythonInteropRuntime.NameFromAddress);
                    using var addr = new PyInt(address);
                    return fromAddress.Invoke(addr);
                }
                finally
                {
                    _type.Dispose();
                }
            }
        }
    }

    /// <summary>
    ///     <c>import weakref</c> — the export keep-alive root. Call under the GIL.
    /// </summary>
    internal static class weakref
    {
        /// <summary>
        ///     <c>weakref.finalize(obj, func)</c> — CPython keeps the finalize object registered until
        ///     the target dies (or interpreter finalization), then calls <paramref name="func"/> once.
        /// </summary>
        internal static PyObject finalize(PyObject obj, PyObject func)
            => PythonInteropRuntime.WeakrefFinalize.Invoke(obj, func);
    }

    /// <summary>
    ///     <c>import builtins</c> — the buffer-protocol front door. Call under the GIL.
    /// </summary>
    internal static class builtins
    {
        /// <summary><c>memoryview(obj)</c> — raises <c>TypeError</c> for non-exporters (surfaced as
        /// <see cref="PythonException"/>, translated by the import path).</summary>
        internal static PyObject memoryview(PyObject obj)
            => PythonInteropRuntime.BuiltinsMemoryview.Invoke(obj);
    }

    /// <summary>
    ///     Python-shaped instance members on <see cref="PyObject"/> — the attribute reads and method
    ///     calls the conversions make on memoryviews and ndarrays, spelled as they are in Python
    ///     (<c>mv.format</c>, <c>mv.itemsize</c>, <c>mv.shape</c>, <c>mv.c_contiguous</c>,
    ///     <c>flat.reshape(dims)</c>, <c>arr.setflags(write: false)</c>, <c>mv.cast("B")</c>,
    ///     <c>mv.tobytes("C")</c>, <c>obj.__array_interface__</c>). Attribute names are the
    ///     session-cached interned PyStrings; call under the GIL.
    /// </summary>
    internal static class PyObjectPythonic
    {
        extension(PyObject obj)
        {
            /// <summary><c>memoryview.format</c> — the PEP 3118 struct format string.</summary>
            internal string format => NDArrayInterop.GetStr(obj, PythonInteropRuntime.NameFormat);

            /// <summary><c>memoryview.itemsize</c>.</summary>
            internal long itemsize => NDArrayInterop.GetLong(obj, PythonInteropRuntime.NameItemsize);

            /// <summary><c>memoryview.shape</c> as element counts (<c>null</c> when Python reports <c>None</c>).</summary>
            internal long[] shape => NDArrayInterop.GetLongTuple(obj, PythonInteropRuntime.NameShape);

            /// <summary><c>memoryview.c_contiguous</c>.</summary>
            internal bool c_contiguous => NDArrayInterop.GetBool(obj, PythonInteropRuntime.NameCContiguous);

            /// <summary><c>obj.__array_interface__</c> — numpy's array-interface dict (v3).</summary>
            internal PyObject __array_interface__ => obj.GetAttr(PythonInteropRuntime.NameArrayInterface);

            /// <summary><c>ndarray.reshape(shape)</c>.</summary>
            internal PyObject reshape(PyObject shape)
            {
                using PyObject method = obj.GetAttr(PythonInteropRuntime.NameReshape);
                return method.Invoke(shape);
            }

            /// <summary><c>ndarray.setflags(write=...)</c> — the first positional IS <c>write</c>.</summary>
            internal void setflags(bool write)
            {
                using PyObject method = obj.GetAttr(PythonInteropRuntime.NameSetflags);
                using PyObject none = method.Invoke(write ? PythonInteropRuntime.TrueLiteral : PythonInteropRuntime.FalseLiteral);
            }

            /// <summary><c>memoryview.cast(format)</c> — <c>"B"</c> routes through the cached literal.</summary>
            internal PyObject cast(string format)
            {
                using PyObject method = obj.GetAttr(PythonInteropRuntime.NameCast);
                if (format == "B")
                    return method.Invoke(PythonInteropRuntime.StrB);
                using var fmt = new PyString(format);
                return method.Invoke(fmt);
            }

            /// <summary><c>memoryview.tobytes(order)</c> — <c>"C"</c> routes through the cached literal.</summary>
            internal PyObject tobytes(string order)
            {
                using PyObject method = obj.GetAttr(PythonInteropRuntime.NameTobytes);
                if (order == "C")
                    return method.Invoke(PythonInteropRuntime.StrC);
                using var o = new PyString(order);
                return method.Invoke(o);
            }
        }
    }
}
