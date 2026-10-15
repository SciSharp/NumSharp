using System;
using System.Runtime.InteropServices;
using System.Threading;
using Python.Runtime;

namespace NumSharp.Interop.PythonNet
{
    /// <summary>
    ///     NumPy's <c>NpyString_*</c> C API (NEP 55, NumPy 2.0+), reached through the function table NumPy publishes in the
    ///     <c>numpy._core._multiarray_umath._ARRAY_API</c> capsule — the same table a C extension compiled against NumPy
    ///     uses. It is how a <c>StringDType</c> (<c>T</c>) array crosses the bridge: NumPy keeps the text of every element
    ///     longer than 15 bytes in an allocator owned by the array's descriptor, outside the array buffer, so only NumPy's
    ///     own functions can read (<see cref="load"/>) or write (<see cref="pack"/>, <see cref="pack_null"/>) those cells.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///     <b>Why the C API and not a Python route</b> (owner's decision D7, 2026-10-13): a <c>T</c> array has no buffer
    ///     protocol, a <c>tolist()</c> round trip costs one Python <c>str</c> per element (about 10× NumPy's own copy), and a
    ///     detour through <c>U</c> drops trailing NULs and cannot find a <c>None</c> missing value. The C API copies every
    ///     element exactly — embedded and trailing NULs, every NA kind, non-BMP text — at the speed of NumPy's own copy loop,
    ///     and the cells it writes are byte-identical to the ones <c>np.array(list, dtype=StringDType(...))</c> builds.
    ///     </para>
    ///     <para>
    ///     <b>Slots.</b> The indices are NumPy's (<c>numpy/_core/code_generators/numpy_api.py</c>): 313 <c>NpyString_load</c>,
    ///     314 <c>NpyString_pack</c>, 315 <c>NpyString_pack_null</c>, 316 <c>NpyString_acquire_allocator</c>, 318
    ///     <c>NpyString_release_allocator</c>, all <c>MinVersion("2.0")</c> — present in every NumPy that has
    ///     <c>StringDType</c> at all, which is what <see cref="EnsureAvailable"/> checks — and 160 <c>PyArray_GetPtr</c>
    ///     (every NumPy), the array's data address without building <c>__array_interface__</c>.
    ///     </para>
    ///     <para>
    ///     <b>Locking and the GIL.</b> Every call runs under the GIL (the bridge's verbs hold it). A copy acquires the NumPy
    ///     allocator ONCE (<see cref="acquire_allocator"/>) and holds it for the whole element loop, as NumPy's own cast loops do; the
    ///     NumSharp side's allocator is locked inside it. NumSharp code never waits for a NumPy allocator, so the two locks
    ///     cannot be taken in the opposite order.
    ///     </para>
    ///     <para>
    ///     <b>Session.</b> The table address belongs to the NumPy module of the current engine session; it is resolved on
    ///     first use and forgotten at engine shutdown (<see cref="SessionEnded"/>).
    ///     </para>
    /// </remarks>
    internal static unsafe class NumpyStringApi
    {
        /// <summary>NumPy's unpacked string (<c>npy_static_string</c>): the text's byte count and its first byte.</summary>
        [StructLayout(LayoutKind.Sequential)]
        internal struct NpyStaticString
        {
            /// <summary>The UTF-8 byte count.</summary>
            internal nuint size;

            /// <summary>The first text byte (NumPy-owned; valid while the allocator is held and the cell is not written).</summary>
            internal byte* buf;
        }

        /// <summary>The <c>_ARRAY_API</c> slot of <c>NpyString_load</c>.</summary>
        private const int LoadSlot = 313;

        /// <summary>The <c>_ARRAY_API</c> slot of <c>NpyString_pack</c>.</summary>
        private const int PackSlot = 314;

        /// <summary>The <c>_ARRAY_API</c> slot of <c>NpyString_pack_null</c>.</summary>
        private const int PackNullSlot = 315;

        /// <summary>The <c>_ARRAY_API</c> slot of <c>NpyString_acquire_allocator</c>.</summary>
        private const int AcquireSlot = 316;

        /// <summary>The <c>_ARRAY_API</c> slot of <c>NpyString_release_allocator</c>.</summary>
        private const int ReleaseSlot = 318;

        /// <summary>The <c>_ARRAY_API</c> slot of <c>PyArray_GetPtr</c>.</summary>
        private const int GetPtrSlot = 160;

        /// <summary>
        ///     The Python program that hands back the table address: <c>PyCapsule_GetPointer</c> called through a PRIVATE
        ///     <c>ctypes</c> prototype (setting <c>restype</c> on <c>ctypes.pythonapi.PyCapsule_GetPointer</c> itself would
        ///     change it for every other user of that shared function object).
        /// </summary>
        private const string ResolveTableProgram =
            "import ctypes\n" +
            "from numpy._core import _multiarray_umath as _mu\n" +
            "_get = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object, ctypes.c_char_p)(('PyCapsule_GetPointer', ctypes.pythonapi))\n" +
            "table = _get(_mu._ARRAY_API, None)\n";

        /// <summary>The <c>_ARRAY_API</c> table of the current engine session (0 until resolved).</summary>
        private static IntPtr _table;

        /// <summary>The resolution gate (the GIL already serializes resolvers; this keeps the six slots consistent).</summary>
        private static readonly object _gate = new object();

        /// <summary><c>int NpyString_load(npy_string_allocator*, const npy_packed_static_string*, npy_static_string*)</c>.</summary>
        /// <remarks>
        ///     Called once per long element of an import, so it skips the GC transition (<c>SuppressGCTransition</c>): the
        ///     function only reads the cell and the arena pointer — no lock, no allocation, no callback, a few nanoseconds —
        ///     which is the contract that attribute requires, and the transition would cost about as much as the call. The
        ///     other entry points keep it: pack and pack_null can allocate or free (malloc takes locks), and acquire waits on
        ///     the allocator's mutex.
        /// </remarks>
        private static delegate* unmanaged[Cdecl, SuppressGCTransition]<IntPtr, byte*, NpyStaticString*, int> _load;

        /// <summary><c>int NpyString_pack(npy_string_allocator*, npy_packed_static_string*, const char*, size_t)</c>.</summary>
        private static delegate* unmanaged[Cdecl]<IntPtr, byte*, byte*, nuint, int> _pack;

        /// <summary><c>int NpyString_pack_null(npy_string_allocator*, npy_packed_static_string*)</c>.</summary>
        private static delegate* unmanaged[Cdecl]<IntPtr, byte*, int> _packNull;

        /// <summary><c>npy_string_allocator* NpyString_acquire_allocator(const PyArray_StringDTypeObject*)</c>.</summary>
        private static delegate* unmanaged[Cdecl]<IntPtr, IntPtr> _acquire;

        /// <summary><c>void NpyString_release_allocator(npy_string_allocator*)</c>.</summary>
        private static delegate* unmanaged[Cdecl]<IntPtr, void> _release;

        /// <summary><c>void* PyArray_GetPtr(PyArrayObject*, const npy_intp*)</c>.</summary>
        private static delegate* unmanaged[Cdecl]<IntPtr, nint*, byte*> _getPtr;

        /// <summary>
        ///     Resolves the table for this engine session when it is not resolved yet, after checking that the running
        ///     NumPy has <c>StringDType</c>. Call under the GIL.
        /// </summary>
        /// <exception cref="NotSupportedException">The running NumPy predates <c>StringDType</c> (NumPy 1.x).</exception>
        /// <exception cref="PythonException">The capsule could not be read (a NumPy without the 2.x module layout).</exception>
        internal static void EnsureAvailable()
        {
            if (Volatile.Read(ref _table) != IntPtr.Zero)
                return;
            lock (_gate)
            {
                if (_table != IntPtr.Zero)
                    return;
                // StringDType is the only reason to reach this table, and every NumPy that has the class has the slots.
                using (PyObject dtypes = PythonRuntimeInterop.Numpy.GetAttr("dtypes"))
                {
                    if (!dtypes.HasAttr("StringDType"))
                        throw new NotSupportedException(
                            "the running NumPy has no StringDType (NEP 55 arrived in NumPy 2.0), so a NumSharp StringDType array cannot cross to it. " +
                            "Upgrade NumPy, or convert first: nd.astype(np.dtype(\"U<n>\")) crosses as a fixed-width U array.");
                }

                IntPtr table;
                using (PyModule scope = Py.CreateScope())
                {
                    scope.Exec(ResolveTableProgram);
                    using PyObject address = scope.Get("table");
                    table = new IntPtr(address.As<long>());
                }

                void** slots = (void**)table;
                _load = (delegate* unmanaged[Cdecl, SuppressGCTransition]<IntPtr, byte*, NpyStaticString*, int>)slots[LoadSlot];
                _pack = (delegate* unmanaged[Cdecl]<IntPtr, byte*, byte*, nuint, int>)slots[PackSlot];
                _packNull = (delegate* unmanaged[Cdecl]<IntPtr, byte*, int>)slots[PackNullSlot];
                _acquire = (delegate* unmanaged[Cdecl]<IntPtr, IntPtr>)slots[AcquireSlot];
                _release = (delegate* unmanaged[Cdecl]<IntPtr, void>)slots[ReleaseSlot];
                _getPtr = (delegate* unmanaged[Cdecl]<IntPtr, nint*, byte*>)slots[GetPtrSlot];
                // Publish last: a reader that sees the table sees the six slots it was read from.
                Volatile.Write(ref _table, table);
            }
        }

        /// <summary>
        ///     Forgets the table at engine shutdown, so a later session resolves its own (a NumPy loaded by another
        ///     interpreter can live at another address).
        /// </summary>
        internal static void SessionEnded()
        {
            lock (_gate)
            {
                Volatile.Write(ref _table, IntPtr.Zero);
                _load = null;
                _pack = null;
                _packNull = null;
                _acquire = null;
                _release = null;
                _getPtr = null;
            }
        }

        /// <summary>
        ///     The address of an array's first logical element (<c>PyArray_DATA</c>, read through <c>PyArray_GetPtr</c> at
        ///     index zero on every axis) — the value <c>arr.__array_interface__['data'][0]</c> reports, for a few
        ///     nanoseconds instead of the 1.5 µs that builds the whole interface dict.
        /// </summary>
        /// <param name="array">A numpy array (an <c>ndarray</c> or subclass; the caller has checked).</param>
        /// <param name="ndim">Its dimension count: <c>PyArray_GetPtr</c> reads one index per axis.</param>
        /// <returns>The data address (the element at index zero; for a reversed axis that is the element NumPy's data
        ///     pointer addresses, not the lowest address).</returns>
        internal static byte* DataAddress(PyObject array, int ndim)
        {
            // A zero index on every axis adds no stride, so the result is the data pointer itself. Stack space for the index
            // keeps the call allocation-free; an array never has more axes than NumPy's NPY_MAXDIMS (64).
            nint* zeros = stackalloc nint[Math.Max(ndim, 1)];
            for (int i = 0; i < ndim; i++)
                zeros[i] = 0;
#pragma warning disable CS0612
            IntPtr arrayObject = array.Handle;
#pragma warning restore CS0612
            return _getPtr(arrayObject, zeros);
        }

        /// <summary>
        ///     <c>NpyString_acquire_allocator</c>: locks the allocator of the array whose descriptor is
        ///     <paramref name="stringDType"/> and returns it. Pair with exactly one <see cref="release_allocator"/>.
        /// </summary>
        /// <param name="stringDType">The array's own descriptor object (<c>arr.dtype</c>) — each NumPy <c>T</c> array has a
        ///     descriptor instance of its own, and views share it; the descriptor passed to a constructor is NOT the new
        ///     array's.</param>
        /// <returns>The locked allocator.</returns>
        internal static IntPtr acquire_allocator(PyObject stringDType)
        {
            // pythonnet 3.x marks Handle obsolete (it steers callers to its internal reference types) but keeps it public
            // through the whole [3.0.5, 4.0.0) range this package accepts; it is the object's address, exactly what the C
            // API takes. Python's id() would give the same number at the cost of a Python call per crossing.
#pragma warning disable CS0612
            IntPtr descriptor = stringDType.Handle;
#pragma warning restore CS0612
            return _acquire(descriptor);
        }

        /// <summary><c>NpyString_release_allocator</c>: unlocks an allocator <see cref="acquire_allocator"/> returned.</summary>
        /// <param name="allocator">The allocator.</param>
        internal static void release_allocator(IntPtr allocator) => _release(allocator);

        /// <summary>
        ///     <c>NpyString_load</c>: the text of a NumPy cell (the allocator must be held).
        /// </summary>
        /// <param name="allocator">The cell's array's allocator.</param>
        /// <param name="cell">The packed cell.</param>
        /// <param name="unpacked">Receives the text pointer and byte count.</param>
        /// <returns>0 for a string, 1 for an NA value, -1 for a cell NumPy cannot read (NumPy sets no Python error).</returns>
        internal static int load(IntPtr allocator, byte* cell, NpyStaticString* unpacked) => _load(allocator, cell, unpacked);

        /// <summary>
        ///     <c>NpyString_pack</c>: frees a NumPy cell's old text and stores <paramref name="size"/> UTF-8 bytes in it (the
        ///     allocator must be held).
        /// </summary>
        /// <param name="allocator">The cell's array's allocator.</param>
        /// <param name="cell">The packed cell.</param>
        /// <param name="text">The UTF-8 text.</param>
        /// <param name="size">Its byte count.</param>
        /// <returns>0, or -1 when NumPy's allocation failed (NumPy sets no Python error).</returns>
        internal static int pack(IntPtr allocator, byte* cell, byte* text, long size) => _pack(allocator, cell, text, (nuint)size);

        /// <summary><c>NpyString_pack_null</c>: makes a NumPy cell an NA value (the allocator must be held).</summary>
        /// <param name="allocator">The cell's array's allocator.</param>
        /// <param name="cell">The packed cell.</param>
        /// <returns>0, or -1 when NumPy could not free the old text.</returns>
        internal static int pack_null(IntPtr allocator, byte* cell) => _packNull(allocator, cell);
    }
}
