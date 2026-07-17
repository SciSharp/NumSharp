using System;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;
using Python.Runtime;

namespace NumSharp.Interop
{
    /// <summary>
    ///     Converts between NumSharp <see cref="NDArray"/> and Python objects via Python.NET (pythonnet).
    ///
    ///     Binds only to pythonnet's <see cref="PyObject"/> and Python's built-in <c>memoryview</c> (the PEP
    ///     3118 buffer protocol) — NO Numpy.NET dependency, so it works with any numpy, any Python, and with
    ///     ANY buffer-protocol exporter (numpy, memoryview, array.array, PIL, torch, ...), not just numpy.
    ///
    ///     NOTE: pythonnet 3.0.1's own <see cref="PyBuffer"/> is broken for shape/format flags
    ///     (PyBUF.ND/STRIDES throw ArgumentOutOfRangeException; PyBUF.FORMATS access-violates). We therefore
    ///     read metadata + C-order bytes through Python's <c>memoryview</c> (which is correct on every
    ///     version), and only use <see cref="PyBuffer"/> with the crash-free <see cref="PyBUF.SIMPLE"/> flag
    ///     to obtain the raw pointer for the zero-copy view.
    ///
    ///     Threading: every method acquires the GIL itself (re-entrant, so nesting under an outer
    ///     <see cref="Py.GIL"/> is fine). The Python engine must be initialized first.
    /// </summary>
    public static class PythonConvert
    {
        // ===========================  NumSharp  ->  Python (numpy)  ===========================

        /// <summary>
        ///     Wrap a NumSharp array as a numpy array that SHARES its unmanaged buffer (mutations visible both
        ///     ways). The NumSharp source is rooted for the lifetime of the returned object. A non-contiguous
        ///     source is materialized C-contiguous first. Requires numpy to be importable.
        /// </summary>
        public static unsafe PyObject ToNumpy(NDArray source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));
            NDArray ns = source.Shape.IsContiguous ? source : np.ascontiguousarray(source);
            string dtypeStr = ToNumpyDtypeStr(ns.typecode);
            long ptr = (long)ns.Storage.Address;
            long nbytes = (long)ns.size * ns.dtypesize;

            using (Py.GIL())
            {
                dynamic ctypes = Py.Import("ctypes");
                dynamic numpy = Py.Import("numpy");
                dynamic flat = numpy.frombuffer((ctypes.c_uint8 * nbytes).from_address(ptr), dtypeStr);
                dynamic arr = ns.ndim == 1
                    ? flat
                    : flat.reshape(new PyTuple(ns.shape.Select(x => (PyObject)new PyInt(x)).ToArray()));
                var py = (PyObject)arr;
                Lifetime.Root(py, ns);
                return py;
            }
        }

        // ===========================  Python  ->  NumSharp  ==================================

        /// <summary>
        ///     Copy any PEP 3118 buffer object (numpy array, memoryview, array.array, ...) into a fresh
        ///     C-contiguous NumSharp array. Honors strides / Fortran order (materialized by CPython's
        ///     <c>memoryview.tobytes('C')</c>). Robust, numpy-agnostic default.
        /// </summary>
        public static NDArray ToNDArray(PyObject obj)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            using (Py.GIL())
            {
                using PyObject mv = Py.Import("builtins").InvokeMethod("memoryview", obj);
                NPTypeCode tc = FromBufferFormat(GetStr(mv, "format"), GetLong(mv, "itemsize"));
                int[] shape = GetShape(mv);
                using PyObject bytes = mv.InvokeMethod("tobytes", "C".ToPython());   // C-order copy
                var nd = np.frombuffer(bytes.As<byte[]>(), tc);
                return shape.Length > 1 ? nd.reshape(shape) : nd;
            }
        }

        /// <summary>
        ///     Zero-copy view of a C-contiguous buffer object: NumSharp SHARES the exporter's memory
        ///     (mutations visible both ways). The pointer comes from a crash-free <see cref="PyBUF.SIMPLE"/>
        ///     buffer, which is rooted to the returned array (keeping the exporter alive) and released when it
        ///     is collected. Non-contiguous inputs are rejected — use <see cref="ToNDArray"/>.
        /// </summary>
        public static unsafe NDArray ToNDArrayView(PyObject obj)
        {
            if (obj is null) throw new ArgumentNullException(nameof(obj));
            using (Py.GIL())
            {
                NPTypeCode tc; int[] dims;
                using (PyObject mv = Py.Import("builtins").InvokeMethod("memoryview", obj))
                {
                    if (!GetBool(mv, "c_contiguous"))
                        throw new InvalidOperationException("buffer is not C-contiguous; use ToNDArray (copy).");
                    tc = FromBufferFormat(GetStr(mv, "format"), GetLong(mv, "itemsize"));
                    dims = GetShape(mv);
                }
                if (dims.Length == 0) dims = new[] { 1 };
                long count = dims.Aggregate(1L, (a, d) => a * d);

                PyBuffer buf = obj.GetBuffer(PyBUF.SIMPLE);   // SIMPLE is the only crash-free flag on pythonnet 3.0.1
                try
                {
                    IArraySlice slice = WrapNonOwning(tc, (void*)buf.Buffer, count);
                    var nd = new NDArray(new UnmanagedStorage(slice, new Shape(dims)));
                    Lifetime.Root(nd, buf);   // buf holds a ref to the exporter; its finalizer releases the view
                    return nd;
                }
                catch
                {
                    buf.Dispose();
                    throw;
                }
            }
        }

        // ---- memoryview metadata helpers (avoid pythonnet 3.0.1's broken PyBuffer) -----------
        static string GetStr(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<string>(); }
        static long GetLong(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<long>(); }
        static bool GetBool(PyObject o, string attr) { using var a = o.GetAttr(attr); return a.As<bool>(); }
        static int[] GetShape(PyObject mv)
        {
            using PyObject shp = mv.GetAttr("shape");        // a tuple of ints
            using var tup = PyTuple.AsTuple(shp);
            int n = (int)tup.Length();
            var dims = new int[n];
            for (int i = 0; i < n; i++) { using var e = tup[i]; dims[i] = checked((int)e.As<long>()); }
            return dims;
        }

        // ===========================  dtype maps  ============================================

        /// <summary>NumSharp dtype -> numpy dtype string ("&lt;i4", "&lt;f8", "|b1", ...).</summary>
        public static string ToNumpyDtypeStr(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean => "|b1", NPTypeCode.Byte => "|u1",  NPTypeCode.SByte => "|i1",
            NPTypeCode.Int16 => "<i2",   NPTypeCode.UInt16 => "<u2", NPTypeCode.Int32 => "<i4",
            NPTypeCode.UInt32 => "<u4",  NPTypeCode.Int64 => "<i8",  NPTypeCode.UInt64 => "<u8",
            NPTypeCode.Half => "<f2",    NPTypeCode.Single => "<f4", NPTypeCode.Double => "<f8",
            NPTypeCode.Complex => "<c16",
            NPTypeCode.Char => "<u2",    // C# char is a 2-byte UTF-16 code unit (numpy has no native char)
            NPTypeCode.Decimal => throw new NotSupportedException(
                "decimal has no numpy dtype (16-byte, non-IEEE). Convert first: nd.astype(NPTypeCode.Double)."),
            _ => throw new NotSupportedException(tc.ToString())
        };

        /// <summary>
        ///     PEP 3118 struct format code (+ itemsize to disambiguate 'l'/'i' across platforms) -> NumSharp
        ///     dtype. An empty format is raw bytes.
        /// </summary>
        public static NPTypeCode FromBufferFormat(string format, long itemSize)
        {
            if (string.IsNullOrEmpty(format))
                return NPTypeCode.Byte;
            int i = "<>=@!".IndexOf(format[0]) >= 0 ? 1 : 0;
            string code = format.Substring(i);
            switch (code)
            {
                case "?": return NPTypeCode.Boolean;
                case "b": return NPTypeCode.SByte;
                case "c": case "B": return NPTypeCode.Byte;
                case "h": return NPTypeCode.Int16;
                case "H": return NPTypeCode.UInt16;
                case "i": case "l": return itemSize == 8 ? NPTypeCode.Int64 : NPTypeCode.Int32;
                case "I": case "L": return itemSize == 8 ? NPTypeCode.UInt64 : NPTypeCode.UInt32;
                case "n": case "q": return NPTypeCode.Int64;
                case "N": case "Q": return NPTypeCode.UInt64;
                case "e": return NPTypeCode.Half;
                case "f": return NPTypeCode.Single;
                case "d": return NPTypeCode.Double;
                case "Zd": return NPTypeCode.Complex;         // complex128
                default:
                    throw new NotSupportedException($"buffer format '{format}' (itemsize {itemSize}) has no NumSharp dtype.");
            }
        }

        static unsafe IArraySlice WrapNonOwning(NPTypeCode tc, void* p, long n)
        {
            switch (tc)
            {
                case NPTypeCode.Boolean: return new ArraySlice<bool>(new UnmanagedMemoryBlock<bool>((bool*)p, n, () => { }));
                case NPTypeCode.Byte:    return new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)p, n, () => { }));
                case NPTypeCode.SByte:   return new ArraySlice<sbyte>(new UnmanagedMemoryBlock<sbyte>((sbyte*)p, n, () => { }));
                case NPTypeCode.Int16:   return new ArraySlice<short>(new UnmanagedMemoryBlock<short>((short*)p, n, () => { }));
                case NPTypeCode.UInt16:  return new ArraySlice<ushort>(new UnmanagedMemoryBlock<ushort>((ushort*)p, n, () => { }));
                case NPTypeCode.Int32:   return new ArraySlice<int>(new UnmanagedMemoryBlock<int>((int*)p, n, () => { }));
                case NPTypeCode.UInt32:  return new ArraySlice<uint>(new UnmanagedMemoryBlock<uint>((uint*)p, n, () => { }));
                case NPTypeCode.Int64:   return new ArraySlice<long>(new UnmanagedMemoryBlock<long>((long*)p, n, () => { }));
                case NPTypeCode.UInt64:  return new ArraySlice<ulong>(new UnmanagedMemoryBlock<ulong>((ulong*)p, n, () => { }));
                case NPTypeCode.Half:    return new ArraySlice<Half>(new UnmanagedMemoryBlock<Half>((Half*)p, n, () => { }));
                case NPTypeCode.Single:  return new ArraySlice<float>(new UnmanagedMemoryBlock<float>((float*)p, n, () => { }));
                case NPTypeCode.Double:  return new ArraySlice<double>(new UnmanagedMemoryBlock<double>((double*)p, n, () => { }));
                case NPTypeCode.Complex: return new ArraySlice<System.Numerics.Complex>(new UnmanagedMemoryBlock<System.Numerics.Complex>((System.Numerics.Complex*)p, n, () => { }));
                default: throw new NotSupportedException(tc.ToString());
            }
        }
    }

    /// <summary>Roots a source object for as long as the bridged object is alive (shared-buffer lifetime).</summary>
    static class Lifetime
    {
        static readonly ConditionalWeakTable<object, object> _roots = new();
        public static void Root(object bridged, object source) => _roots.AddOrUpdate(bridged, source);
    }
}
