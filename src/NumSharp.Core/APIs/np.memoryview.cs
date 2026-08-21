using System;
using System.Runtime.CompilerServices;

namespace NumSharp
{
    public static partial class np
    {
        // There is deliberately NO `np.memoryview(a)` factory: `memoryview` is a Python builtin,
        // not a member of NumPy's `np.*` namespace. NumPy exposes the buffer object ONLY through the
        // `ndarray.data` attribute, so NumSharp mirrors that — a `MemoryView` is obtained solely from
        // <see cref="NDArray.data"/>.

        /// <summary>
        ///     The NumSharp analog of Python's <c>memoryview</c> — the buffer object returned by
        ///     <see cref="NDArray.data"/>. It is a lightweight, zero-copy HANDLE onto an array's raw
        ///     memory plus the layout metadata needed to interpret it (NumPy's <c>ndarray.data</c> is
        ///     literally <c>memoryview(self)</c>).
        ///
        ///     <para>
        ///     It owns no memory of its own: every member reads LIVE through the source array's
        ///     <see cref="NDArray.Storage"/> and <see cref="NDArray.Shape"/>, so it stays valid while
        ///     the source array (held as <see cref="obj"/>, keeping it alive exactly as NumPy's
        ///     <c>memoryview.obj</c> does) is alive and not structurally mutated. The
        ///     <see cref="Pointer"/> addresses the LOGICAL first element (<c>base + offset·itemsize</c>),
        ///     matching NumPy's <c>a.data</c> / <c>PyArray_DATA</c> / <c>a.ctypes.data</c> /
        ///     <c>a.__array_interface__['data'][0]</c> — for a sliced or reversed view this is the offset
        ///     element, not the buffer base.
        ///     </para>
        ///
        ///     <para>
        ///     Surface (probed against NumPy 2.4.2's <c>memoryview</c>): <see cref="obj"/>,
        ///     <see cref="nbytes"/>, <see cref="itemsize"/>, <see cref="ndim"/>, <see cref="readonly"/>
        ///     (true for broadcast / non-writeable views), <see cref="shape"/>, <see cref="strides"/>
        ///     (in BYTES like NumPy's — NumSharp's own strides are in elements), <see cref="format"/>
        ///     (the struct-module type code), <see cref="c_contiguous"/> / <see cref="f_contiguous"/> /
        ///     <see cref="contiguous"/>, <see cref="Length"/> (NumPy's <c>len(mv)</c>),
        ///     <see cref="tobytes(string)"/> / <see cref="hex()"/>, the raw <see cref="Pointer"/> /
        ///     <see cref="Address"/>, and write-through scalar element access via
        ///     <see cref="this[long[]]"/>.
        ///     </para>
        ///
        ///     <para>
        ///     Deliberately NOT modelled (documented boundaries, not gaps to close): partial-index
        ///     sub-views (<c>mv[1]</c> on an N-D memoryview yields a sub-<c>memoryview</c> in Python —
        ///     use the <see cref="NDArray"/> indexer for sub-arrays), <c>cast</c>, <c>tolist</c>,
        ///     <c>release</c> / the context-manager protocol, and element iteration. NumSharp has no
        ///     Python buffer protocol, so these Python-object conveniences have no counterpart; the
        ///     buffer ESSENCE (pointer + metadata + write-through + <c>tobytes</c>) is what
        ///     <c>ndarray.data</c> is for.
        ///     </para>
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html</remarks>
        public sealed unsafe class MemoryView
        {
            private readonly NDArray _arr;

            internal MemoryView(NDArray a)
            {
                _arr = a ?? throw new ArgumentNullException(nameof(a));
            }

            /// <summary>
            ///     The underlying array this view exposes (NumPy's <c>memoryview.obj</c>). Holding a
            ///     <see cref="MemoryView"/> keeps this array reachable, mirroring how a Python
            ///     <c>memoryview</c> keeps its exporter alive.
            /// </summary>
            public NDArray obj => _arr;

            /// <summary>Size of one element in bytes (NumPy's <c>memoryview.itemsize</c>).</summary>
            public int itemsize => _arr.dtypesize;

            /// <summary>Number of dimensions (NumPy's <c>memoryview.ndim</c>); <c>0</c> for a scalar array.</summary>
            public int ndim => _arr.Shape.NDim;

            /// <summary>
            ///     Total LOGICAL byte count = <c>product(shape)·itemsize</c> (NumPy's
            ///     <c>memoryview.nbytes</c>). A broadcast view reports its logical size, a 0-d array one
            ///     itemsize, an empty array <c>0</c>.
            /// </summary>
            public long nbytes => _arr.Shape.Size * (long)_arr.dtypesize;

            /// <summary>
            ///     <c>true</c> when the underlying memory may not be written (NumPy's
            ///     <c>memoryview.readonly</c>) — set for broadcast views, which NumSharp (like NumPy)
            ///     marks non-writeable. Named to match NumPy; because <c>readonly</c> is a C# keyword,
            ///     access it as <c>mv.@readonly</c>.
            /// </summary>
            public bool @readonly => !_arr.Shape.IsWriteable;

            /// <summary>The array's shape (NumPy's <c>memoryview.shape</c>). A fresh array per read.</summary>
            public long[] shape
            {
                get
                {
                    var dims = _arr.Shape.Dimensions;
                    var ret = new long[dims.Length];
                    Array.Copy(dims, ret, dims.Length);
                    return ret;
                }
            }

            /// <summary>
            ///     Strides in BYTES (NumPy's <c>memoryview.strides</c>). NumSharp stores strides in
            ///     ELEMENTS, so each is scaled by <see cref="itemsize"/> here — a broadcast axis keeps its
            ///     <c>0</c> stride, a reversed view its negative stride. A fresh array per read.
            /// </summary>
            public long[] strides
            {
                get
                {
                    var st = _arr.Shape.Strides;
                    long isz = _arr.dtypesize;
                    var ret = new long[st.Length];
                    for (int i = 0; i < st.Length; i++)
                        ret[i] = st[i] * isz;
                    return ret;
                }
            }

            /// <summary>
            ///     The struct-module type code describing one element (NumPy's <c>memoryview.format</c>),
            ///     e.g. <c>"d"</c> for float64, <c>"?"</c> for bool, <c>"Zd"</c> for complex128. The 32-bit
            ///     integer codes are platform-dependent exactly as NumPy's are (its C <c>long</c> maps to
            ///     <c>'l'</c>/<c>'L'</c> on Windows LLP64 but <c>'i'</c>/<c>'I'</c> on LP64), because
            ///     NumPy's int32 dtype is <c>NPY_LONG</c> on Windows and <c>NPY_INT</c> elsewhere. The two
            ///     NumSharp-only dtypes that have NO NumPy analog carry a documented NumSharp-specific
            ///     code: <see cref="NPTypeCode.Char"/> (2-byte UTF-16) → <c>"u"</c>,
            ///     <see cref="NPTypeCode.Decimal"/> (opaque 16-byte item) → <c>"16s"</c>.
            /// </summary>
            public string format => FormatCode(_arr.typecode);

            /// <summary><c>true</c> if the array is C-contiguous (NumPy's <c>memoryview.c_contiguous</c>).</summary>
            public bool c_contiguous => _arr.Shape.IsContiguous;

            /// <summary><c>true</c> if the array is Fortran-contiguous (NumPy's <c>memoryview.f_contiguous</c>).</summary>
            public bool f_contiguous => _arr.Shape.IsFContiguous;

            /// <summary>
            ///     <c>true</c> if the array is contiguous in EITHER C or Fortran order (NumPy's
            ///     <c>memoryview.contiguous</c> ≡ <c>PyBuffer_IsContiguous(view, 'A')</c>).
            /// </summary>
            public bool contiguous => _arr.Shape.IsContiguous || _arr.Shape.IsFContiguous;

            /// <summary>
            ///     Raw pointer to the LOGICAL first byte of the array's data
            ///     (<c>Storage.Address + offset·itemsize</c>) — the C-level analog of NumPy's
            ///     <c>a.data</c> pointer / <c>PyArray_DATA(self)</c>. Reads and writes through it hit the
            ///     underlying buffer directly (subject to <see cref="readonly"/>).
            /// </summary>
            public void* Pointer => (byte*)_arr.Storage.Address + _arr.Shape.Offset * _arr.dtypesize;

            /// <summary>
            ///     <see cref="Pointer"/> as an <see cref="IntPtr"/> — equal to NumPy's
            ///     <c>a.ctypes.data</c> / <c>a.__array_interface__['data'][0]</c>. Convenient for P/Invoke
            ///     and native interop.
            /// </summary>
            public IntPtr Address => (IntPtr)Pointer;

            /// <summary>
            ///     The size of the FIRST dimension (NumPy's <c>len(memoryview)</c>). Throws for a 0-d
            ///     array, matching NumPy's <c>TypeError: len() of unsized object</c>.
            /// </summary>
            public long Length => ndim == 0
                ? throw new InvalidOperationException("len() of unsized object")
                : _arr.Shape.Dimensions[0];

            /// <summary>
            ///     Write-through scalar element access (NumPy's <c>memoryview</c> full-index element get/set).
            ///     Supply exactly one index per dimension (a 0-d array takes a length-0 index array);
            ///     negative indices count from the end. Reads/writes go through the array's stride-aware
            ///     accessors, so they honour any layout and write THROUGH to the base — a strided /
            ///     transposed / reversed / offset view included. Partial indexing (fewer indices than
            ///     <see cref="ndim"/>, which yields a sub-<c>memoryview</c> in Python) is not modelled;
            ///     use the <see cref="NDArray"/> indexer for sub-arrays.
            /// </summary>
            /// <param name="indices">One index per dimension (length must equal <see cref="ndim"/>).</param>
            /// <exception cref="ArgumentException">If <paramref name="indices"/>'s length does not equal <see cref="ndim"/>.</exception>
            /// <exception cref="IndexOutOfRangeException">If any (normalized) index is out of bounds.</exception>
            /// <exception cref="InvalidOperationException">On set, if this view is <see cref="readonly"/>.</exception>
            public object this[params long[] indices]
            {
                get => _arr.GetValue(ResolveCoords(indices));
                set
                {
                    if (@readonly)
                        throw new InvalidOperationException("cannot modify read-only memory");
                    _arr.SetValue(CoerceToDtype(value), ResolveCoords(indices));
                }
            }

            // The array's per-index setter demands the boxed value's CLR type match the dtype EXACTLY, so
            // (like NumPy's memoryview, which casts an assigned value to the buffer's format) coerce first.
            // A same-typed value passes straight through; anything else rides a 1-element astype — the same
            // scalar-cast path NumSharp uses everywhere, so it covers all 15 dtypes (Half/Complex/Decimal
            // included). NOTE: astype WRAPS an out-of-range integer (NumSharp's strong-cast rule) where
            // NumPy's memoryview raises; assign through NDArray.flatiter for the weak-scalar bounds check.
            private object CoerceToDtype(object value)
            {
                if (value is null)
                    throw new ArgumentNullException(nameof(value));
                if (value.GetType() == _arr.dtype)
                    return value;
                var src = new NDArray(value.GetType(), 1);
                src.SetAtIndex(value, 0);
                return src.astype(_arr.dtype).GetAtIndex(0);
            }

            // Validate arity, normalize negatives, and bounds-check — mirrors memoryview's IndexError contract.
            private long[] ResolveCoords(long[] indices)
            {
                indices ??= Array.Empty<long>();
                int nd = ndim;
                if (indices.Length != nd)
                    throw new ArgumentException($"cannot index {nd}-dimensional memory with {indices.Length} indices");

                var dims = _arr.Shape.Dimensions;
                var coords = new long[nd];
                for (int i = 0; i < nd; i++)
                {
                    long ix = indices[i];
                    long dim = dims[i];
                    if (ix < 0)
                        ix += dim;
                    if (ix < 0 || ix >= dim)
                        throw new IndexOutOfRangeException($"index out of bounds on dimension {i + 1}");
                    coords[i] = ix;
                }
                return coords;
            }

            /// <summary>
            ///     Copies the array's elements into a fresh <c>byte[]</c> in the requested logical order
            ///     (NumPy's <c>memoryview.tobytes(order='C')</c>): <c>'C'</c> row-major (default),
            ///     <c>'F'</c> column-major, <c>'A'</c> = the physical order if the array is already
            ///     contiguous (Fortran order when Fortran-contiguous, else C). Follows strides, so a
            ///     strided / reversed / transposed view is materialized in logical order — matching
            ///     <c>bytes(a.data)</c>.
            /// </summary>
            /// <param name="order"><c>"C"</c> (default), <c>"F"</c>, or <c>"A"</c>.</param>
            /// <exception cref="ArgumentException">If <paramref name="order"/> is not one of C/F/A.</exception>
            /// <exception cref="NotSupportedException">If the byte count exceeds <see cref="int.MaxValue"/> (a <c>byte[]</c> limit, not a format one).</exception>
            public byte[] tobytes(string order = "C")
            {
                char o = ResolveOrder(order);
                long total = nbytes;
                if (total > int.MaxValue)
                    throw new NotSupportedException($"tobytes(): result of {total} bytes exceeds the maximum byte[] length ({int.MaxValue}).");

                var ret = new byte[total];
                if (total == 0)
                    return ret;

                // 'A' means: keep the physical layout if already contiguous, else fall back to C-order.
                char effective = o == 'A' ? (f_contiguous && !c_contiguous ? 'F' : 'C') : o;

                // Read the bytes contiguously off an array whose buffer, from its logical start, already
                // holds the elements in `effective` order. If we're already contiguous in that order,
                // read directly off Pointer; otherwise materialize a contiguous copy through the existing
                // copy machinery (which drives NDIter / the IL copy kernels internally).
                bool alreadyOrdered = (effective == 'C' && c_contiguous) || (effective == 'F' && f_contiguous);
                NDArray src = alreadyOrdered ? _arr : _arr.copy(effective);
                byte* p = (byte*)src.Storage.Address + src.Shape.Offset * src.dtypesize;

                fixed (byte* dst = ret)
                    Buffer.MemoryCopy(p, dst, total, total);

                return ret;
            }

            /// <summary>
            ///     The array's bytes as a lower-case hex string (NumPy's <c>memoryview.hex()</c>), in
            ///     logical C-order. Equivalent to hex-encoding <see cref="tobytes()"/>.
            /// </summary>
            public string hex() => Convert.ToHexString(tobytes("C")).ToLowerInvariant();

            private static char ResolveOrder(string order)
            {
                if (order is null || order.Length != 1)
                    throw new ArgumentException("order must be one of 'C', 'F', or 'A'");
                char o = order[0];
                if (o != 'C' && o != 'F' && o != 'A')
                    throw new ArgumentException("order must be one of 'C', 'F', or 'A'");
                return o;
            }

            // NumPy's dtype -> struct format code (numpy/_core/src/multiarray/buffer.c). NPY_LONG emits
            // 'q' on LP64 (sizeof(long)==sizeof(longlong)) but 'l' on Windows LLP64 — and NumPy's int32
            // dtype is NPY_LONG on Windows / NPY_INT elsewhere — so int32/uint32 are the only
            // platform-dependent codes. Char and Decimal have no NumPy dtype (documented house codes).
            private static bool CLongIs32Bit => OperatingSystem.IsWindows();

            private static string FormatCode(NPTypeCode tc)
            {
                switch (tc)
                {
                    case NPTypeCode.Boolean: return "?";
                    case NPTypeCode.SByte: return "b";
                    case NPTypeCode.Byte: return "B";
                    case NPTypeCode.Int16: return "h";
                    case NPTypeCode.UInt16: return "H";
                    case NPTypeCode.Int32: return CLongIs32Bit ? "l" : "i";
                    case NPTypeCode.UInt32: return CLongIs32Bit ? "L" : "I";
                    case NPTypeCode.Int64: return "q";
                    case NPTypeCode.UInt64: return "Q";
                    case NPTypeCode.Half: return "e";
                    case NPTypeCode.Single: return "f";
                    case NPTypeCode.Double: return "d";
                    case NPTypeCode.Complex: return "Zd";
                    case NPTypeCode.Char: return "u";    // NumSharp-only: 2-byte UTF-16 code unit (no NumPy dtype)
                    case NPTypeCode.Decimal: return "16s"; // NumSharp-only: opaque 16-byte item (no NumPy dtype)
                    default: throw new NotSupportedException($"No memoryview format code for {tc}.");
                }
            }

            /// <summary>An informative debug string (NumPy's <c>memoryview</c> repr is the opaque <c>&lt;memory at 0x…&gt;</c>).</summary>
            public override string ToString()
                => $"<NumSharp.MemoryView shape=[{string.Join(",", _arr.Shape.Dimensions)}] format='{format}' readonly={@readonly}>";
        }
    }

    public partial class NDArray
    {
        /// <summary>
        ///     Python buffer object pointing to the start of the array's data — the NumSharp analog of
        ///     NumPy's <c>ndarray.data</c> (which is literally <c>memoryview(self)</c>). Returns a
        ///     zero-copy <see cref="np.MemoryView"/> handle over this array's memory: it exposes the raw
        ///     <see cref="np.MemoryView.Pointer"/> at the LOGICAL first element (so a sliced or reversed
        ///     view reports its offset element, matching NumPy's <c>a.data</c> / <c>a.ctypes.data</c>),
        ///     the buffer metadata (<c>nbytes</c> / <c>itemsize</c> / <c>ndim</c> / <c>shape</c> /
        ///     <c>strides</c> in bytes / <c>format</c> / <c>readonly</c> / contiguity), write-through
        ///     element access, and <c>tobytes</c>. Read-only, like NumPy's attribute (which raises
        ///     <c>AttributeError</c> on assignment); a fresh handle is returned per access. See
        ///     <see cref="Data{T}"/> / <see cref="GetData()"/> for the typed / raw-slice accessors.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.ndarray.data.html</remarks>
        public np.MemoryView data => new np.MemoryView(this);
    }
}
