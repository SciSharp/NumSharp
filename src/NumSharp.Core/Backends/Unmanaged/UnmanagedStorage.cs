using System;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

// ReSharper disable once CheckNamespace
namespace NumSharp.Backends
{
    /// <summary>
    ///     Serves as a typed storage for an array.
    /// </summary>
    /// <remarks>
    ///     Responsible for :<br></br>
    ///      - store data type, elements, Shape<br></br>
    ///      - offers methods for accessing elements depending on shape<br></br>
    ///      - offers methods for casting elements<br></br>
    ///      - offers methods for change tensor order<br></br>
    ///      - GetData always return reference object to the true storage<br></br>
    ///      - GetData{T} and SetData{T} change dtype and cast storage<br></br>
    ///      - CloneData always create a clone of storage and return this as reference object<br></br>
    ///      - CloneData{T} clone storage and cast this clone <br></br>
    /// </remarks>
    public partial class UnmanagedStorage : ICloneable
    {
        /// <summary>
        ///     The 15 per-dtype <see cref="ArraySlice{T}"/> lanes overlapped at offset 0 — exactly one
        ///     lane (the one matching <see cref="_typecode"/>) is ever written or read over a storage's
        ///     lifetime, so the union holds the single live typed slice in 64 B instead of 15 separate
        ///     64 B fields (−896 B per storage instance). The overlap is CLR-legal because every lane
        ///     has the identical layout for any T (T occurs only behind pointers: T*/void*/long/bool)
        ///     and the single managed reference — the non-generic <c>Disposer</c> inside
        ///     <see cref="UnmanagedMemoryBlock{T}"/> — sits at the same offset with the same type in
        ///     all lanes, so the GC ref map is well-formed (the type loader rejects the layout
        ///     otherwise). See docs/UNMANAGED_STORAGE_UNION_DESIGN.md.
        /// </summary>
        [StructLayout(LayoutKind.Explicit)]
        private struct TypedSlices
        {
            // %foreach supported_dtypes,supported_dtypes_lowercase%
            // [FieldOffset(0)] public ArraySlice<#2> #1;
            [FieldOffset(0)] public ArraySlice<bool> Boolean;
            [FieldOffset(0)] public ArraySlice<sbyte> SByte;
            [FieldOffset(0)] public ArraySlice<byte> Byte;
            [FieldOffset(0)] public ArraySlice<short> Int16;
            [FieldOffset(0)] public ArraySlice<ushort> UInt16;
            [FieldOffset(0)] public ArraySlice<int> Int32;
            [FieldOffset(0)] public ArraySlice<uint> UInt32;
            [FieldOffset(0)] public ArraySlice<long> Int64;
            [FieldOffset(0)] public ArraySlice<ulong> UInt64;
            [FieldOffset(0)] public ArraySlice<char> Char;
            [FieldOffset(0)] public ArraySlice<Half> Half;
            [FieldOffset(0)] public ArraySlice<double> Double;
            [FieldOffset(0)] public ArraySlice<float> Single;
            [FieldOffset(0)] public ArraySlice<decimal> Decimal;
            [FieldOffset(0)] public ArraySlice<System.Numerics.Complex> Complex;
        }

        private TypedSlices _slices;
        public IArraySlice InternalArray;
        public unsafe byte* Address;
        public long Count;

        protected Type _dtype;
        protected NPTypeCode _typecode;
        protected Shape _shape;

        /// <summary>
        /// The original storage this is a view of, or <c>null</c> if this storage owns its data.
        /// </summary>
        /// <remarks>
        /// <para>
        /// <b>Memory Model:</b> All views chain to the ultimate owner (not intermediate views).
        /// When storage B is a view of A, and C is a view of B, then both B._baseStorage and
        /// C._baseStorage point to A (not C → B → A).
        /// </para>
        /// <para>
        /// <b>Memory Safety:</b> The underlying memory is kept alive by the shared Disposer
        /// class reference in the UnmanagedMemoryBlock. This field provides semantic tracking
        /// for the NumPy-compatible <see cref="NDArray.@base"/> property, not GC safety.
        /// </para>
        /// <para>
        /// <b>Set by:</b> All three <see cref="Alias()"/> overloads,
        /// <see cref="CreateBroadcastedUnsafe(UnmanagedStorage, Shape)"/>,
        /// and both <see cref="GetData(int[])"/> overloads when creating views.
        /// </para>
        /// </remarks>
        /// <seealso cref="BaseStorage"/>
        /// <seealso cref="IsView"/>
        internal UnmanagedStorage? _baseStorage;

        /// <summary>
        /// Gets the original storage this is a view of, or <c>null</c> if this storage owns its data.
        /// </summary>
        /// <value>
        /// The ultimate owner storage for views, or <c>null</c> for owned data.
        /// </value>
        /// <remarks>
        /// <para>
        /// NumPy-compatible: All views chain to the ultimate owner (not intermediate views).
        /// </para>
        /// <para>
        /// <b>Example:</b>
        /// <code>
        /// var a = np.arange(10);           // a.Storage.BaseStorage == null (owns data)
        /// var b = a["2:5"];                // b.Storage.BaseStorage == a.Storage
        /// var c = b["1:2"];                // c.Storage.BaseStorage == a.Storage (chains to original!)
        /// </code>
        /// </para>
        /// <para>
        /// <b>Note:</b> This property is read-only by design. Allowing external modification would
        /// risk breaking the memory ownership chain and could lead to use-after-free bugs.
        /// </para>
        /// </remarks>
        /// <seealso href="https://numpy.org/doc/stable/reference/generated/numpy.ndarray.base.html"/>
        public UnmanagedStorage? BaseStorage => _baseStorage;

        /// <summary>
        /// Gets a value indicating whether this storage is a view of another storage.
        /// </summary>
        /// <value>
        /// <c>true</c> if this storage shares memory with another storage (does not own its data);
        /// <c>false</c> if this storage owns its data.
        /// </value>
        /// <remarks>
        /// <para>
        /// Equivalent to checking <c>BaseStorage != null</c>.
        /// </para>
        /// <para>
        /// <b>Use cases:</b>
        /// <list type="bullet">
        ///   <item>Determine if an array can be safely modified without affecting other arrays</item>
        ///   <item>Optimize copy-on-write patterns</item>
        ///   <item>Debug memory sharing issues</item>
        /// </list>
        /// </para>
        /// </remarks>
        public bool IsView => _baseStorage != null;

        /// <summary>
        ///     Set when this storage wraps FOREIGN memory that must never become writeable — a read-only
        ///     memory-mapped file (<c>np.load(mmap_mode: "r")</c>), or a buffer whose exporter declared it
        ///     read-only (<c>np.frombuffer</c> over a read-only <see cref="MemoryView"/>). It is the gate
        ///     NumPy expresses through the buffer protocol (its <c>_IsWriteable</c> asks the non-array base
        ///     for a writable buffer and a read-only exporter refuses): without it,
        ///     <c>setflags(write: true)</c> on an <c>'r'</c> memmap would re-enable WRITEABLE and the next
        ///     write would reach PROT_READ pages — an access violation, not an exception.
        /// </summary>
        /// <remarks>
        ///     Ordinary owned allocations and ordinary views never set this; a cleared WRITEABLE flag alone
        ///     (e.g. <c>a.flags.writeable = false</c> on your own array) is reversible and does NOT imply it.
        /// </remarks>
        internal bool WriteProtected;

        /// <summary>
        ///     Set when this storage's memory belongs to a FOREIGN owner even though no
        ///     <see cref="_baseStorage"/> exists — a memory-mapped file (every <c>mmap_mode</c>, the file
        ///     mapping is the real owner), or an external buffer wrap. It is the NumPy analog of a
        ///     non-array <c>base</c> object: such arrays report <c>flags.owndata == False</c> (NumPy's
        ///     memmap does, in every mode) and <c>np.require(…, "O")</c> copies them.
        /// </summary>
        /// <remarks>Orthogonal to <see cref="WriteProtected"/>: an <c>'r+'</c> memmap is externally based
        /// yet writable; an <c>'r'</c> memmap is both.</remarks>
        internal bool ExternalBase;

        /// <summary>
        ///     Does this storage own the memory it addresses — NumPy's <c>OWNDATA</c>: no base storage
        ///     (<see cref="_baseStorage"/>) and no foreign owner (<see cref="ExternalBase"/>). Read by
        ///     <c>ndarray.flags.owndata</c> and <c>np.require("O")</c>.
        /// </summary>
        internal bool OwnsData => _baseStorage is null && !ExternalBase;

        /// <summary>
        ///     Post-transition hook — run after this storage's <see cref="_shape"/> is (re)assigned or its
        ///     ownership links (<see cref="_baseStorage"/> / <see cref="ExternalBase"/>) change. Reconciles
        ///     the shape's <see cref="ArrayFlags.OWNDATA"/> bit with <see cref="OwnsData"/> — the NumSharp
        ///     analog of NumPy's <c>PyArray_NewFromDescr_int</c>
        ///     (<c>ctors.c</c>: <c>fa-&gt;flags |= NPY_ARRAY_OWNDATA</c> when it allocates,
        ///     <c>fa-&gt;flags &amp;= ~NPY_ARRAY_OWNDATA</c> when data is passed in) plus the invariant
        ///     NumPy asserts in <c>_IsWriteable</c> (<c>common.c</c>): <c>base != NULL ⟹ !OWNDATA</c>.
        /// </summary>
        /// <remarks>
        ///     Must be called wherever ownership or <see cref="_shape"/> transitions: at the end of every
        ///     allocating constructor / <see cref="_Allocate"/>, after every <see cref="_baseStorage"/>
        ///     assignment (views borrow the parent's shape, which carries the parent's bit), after
        ///     <see cref="ExternalBase"/> is raised, and in <see cref="SetShapeUnsafe(ref Shape)"/> /
        ///     the <c>ReplaceData</c> family (freshly built shapes never carry the bit, which would
        ///     silently strip it from an owner).
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected internal void OnReshaped()
        {
            bool owns = _baseStorage is null && !ExternalBase;
            if (((_shape._flags & (int)ArrayFlags.OWNDATA) != 0) != owns)
                _shape = owns
                    ? _shape.WithFlags(flagsToSet: ArrayFlags.OWNDATA)
                    : _shape.WithFlags(flagsToClear: ArrayFlags.OWNDATA);
        }

        /// <summary>
        ///     May this array's WRITEABLE flag be turned (back) on? The NumSharp analog of NumPy's
        ///     <c>_IsWriteable</c> (<c>numpy/_core/src/multiarray/common.c</c>), consulted by
        ///     <see cref="NDArray.setflags"/> / <c>flags.writeable = true</c>:
        ///     <list type="bullet">
        ///         <item>an array over foreign read-only memory (<see cref="WriteProtected"/>) — never
        ///         (NumPy: the non-array base refuses a writable buffer);</item>
        ///         <item>a view — iff its base is writeable (NumPy walks the collapsed base chain and
        ///         accepts if ANY base is writeable; NumSharp's <see cref="_baseStorage"/> is flattened to
        ///         the ultimate owner, which is that same answer);</item>
        ///         <item>an owner of ordinary memory — always (NumPy: <c>base == NULL || OWNDATA</c>).</item>
        ///     </list>
        /// </summary>
        internal bool CanEnableWriteable()
        {
            if (WriteProtected)
                return false;
            var b = _baseStorage;
            if (b is not null)
                return b.Shape.IsWriteable && !b.WriteProtected;
            return true;
        }

        /// <summary>
        ///     The data type of internal storage array.
        /// </summary>
        /// <value>numpys equal dtype</value>
        /// <remarks>Has to be compliant with <see cref="NPTypeCode"/>.</remarks>
        public Type DType => _dtype;

        /// <summary>
        ///     The <see cref="NPTypeCode"/> of <see cref="IStorage.DType"/>.
        /// </summary>
        public NPTypeCode TypeCode => _typecode;

        /// <summary>
        ///     The size in bytes of a single value of <see cref="DType"/>
        ///     as stored in the unmanaged buffer.
        /// </summary>
        /// <remarks>
        /// Returns the in-memory element stride, not the marshaling size.
        /// For bool that is 1, not <see cref="Marshal.SizeOf(object)"/>'s 4
        /// (bool is marshaled to win32 BOOL = int). All pointer arithmetic
        /// over <c>Address</c> uses this value, so the in-memory layout is
        /// the only correct reference.
        /// </remarks>
        public int DTypeSize
        {
            get
            {
                if (_typecode == NPTypeCode.String)
                {
                    return IntPtr.Size;
                }

                return _typecode.SizeOf();
            }
        }

        /// <summary>
        ///     The shape representing the data in this storage.
        /// </summary>
        public Shape Shape
        {
            get
            {
                return _shape;
            }
            set
            {
                this.Reshape(ref value);
            }
        }

        /// <summary>
        ///     The shape representing the data in this storage.
        /// </summary>
        /// <remarks>It is dangerous to set Shape by reference. use Reshape(Shape) instead.</remarks>
        public ref Shape ShapeReference => ref _shape;

        /// <summary>
        /// Returns an UnmanagedSpan representing this storage's memory.
        /// </summary>
        /// <remarks>This ignores completely slicing. Supports long indexing for arrays &gt; 2B elements.</remarks>
        public unsafe Span<T> AsSpan<T>() where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"AsSpan<{typeof(T).Name}> called on {_dtype.Name} array.");
            if (!_shape.IsContiguous)
                throw new InvalidOperationException("Unable to span a non-contiguous storage.");

            return new Span<T>(Address, (int)Count);
        }

        /// <summary>
        /// Returns an UnmanagedSpan representing this storage's memory.
        /// </summary>
        /// <remarks>This ignores completely slicing. Supports long indexing for arrays &gt; 2B elements.</remarks>
        public unsafe UnmanagedSpan<T> AsUnmanagedSpan<T>() where T : unmanaged
        {
            Debug.Assert(typeof(T) == _dtype, $"AsUnmanagedSpan<{typeof(T).Name}> called on {_dtype.Name} array.");
            if (!_shape.IsContiguous)
                throw new InvalidOperationException("Unable to span a non-contiguous storage.");

            return new UnmanagedSpan<T>(Address, Count);
        }

        /// <summary>
        ///     The engine that was used to create this <see cref="IStorage"/>.
        /// </summary>
        public TensorEngine Engine { get; protected internal set; }

        public static UnmanagedStorage Scalar<T>(T value) where T : unmanaged => new UnmanagedStorage(ArraySlice.Scalar<T>(value));

        public static UnmanagedStorage Scalar(object value) => new UnmanagedStorage(ArraySlice.Scalar(value));

        public static UnmanagedStorage Scalar(object value, NPTypeCode typeCode) => new UnmanagedStorage(ArraySlice.Scalar(value, typeCode));

        /// <summary>
        /// Creates a new storage with a broadcasted shape from an array slice.
        /// </summary>
        /// <param name="arraySlice">The array slice to wrap.</param>
        /// <param name="shape">The broadcasted shape to represent this storage.</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that owns the data (not a view).
        /// The returned storage's <see cref="_baseStorage"/> is <c>null</c>.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Named "Unsafe":</b> This method does not validate that the shape is compatible
        /// with the array slice size.
        /// </para>
        /// <para>
        /// <b>Ownership:</b> This overload creates owned storage (not a view) because it
        /// receives raw data without storage context. Compare with the
        /// <see cref="CreateBroadcastedUnsafe(UnmanagedStorage, Shape)"/> overload which
        /// preserves base tracking.
        /// </para>
        /// </remarks>
        public static UnmanagedStorage CreateBroadcastedUnsafe(IArraySlice arraySlice, Shape shape)
        {
            var ret = new UnmanagedStorage();
            ret._Allocate(shape, arraySlice);
            return ret;
        }

        /// <summary>
        /// Creates a broadcasted view of an existing storage with a new shape.
        /// </summary>
        /// <param name="storage">The source storage to take <see cref="InternalArray"/> from.</param>
        /// <param name="shape">The broadcasted shape to represent this storage.</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with the source storage.
        /// The returned storage's <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Named "Unsafe":</b> This method does not validate that the shape is compatible
        /// with the storage size.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner:
        /// <list type="bullet">
        ///   <item>If source storage owns its data: <c>result._baseStorage = storage</c></item>
        ///   <item>If source storage is a view: <c>result._baseStorage = storage._baseStorage</c></item>
        /// </list>
        /// </para>
        /// <para>
        /// <b>Used By:</b> <c>np.broadcast_to()</c>, <c>np.broadcast_arrays()</c>, and
        /// internal broadcasting operations.
        /// </para>
        /// </remarks>
        /// <seealso cref="Alias()"/>
        public static UnmanagedStorage CreateBroadcastedUnsafe(UnmanagedStorage storage, Shape shape)
        {
            var ret = new UnmanagedStorage();
            ret._Allocate(shape, storage.InternalArray);
            ret._baseStorage = storage._baseStorage ?? storage;
            ret.OnReshaped(); // a view never owns its buffer (NumPy: base != NULL ⟹ !OWNDATA)
            ret.Engine = storage.Engine;
            return ret;
        }


        private UnmanagedStorage() { }

        /// <summary>
        ///     <see cref="Shape.Scalar"/> with <see cref="ArrayFlags.OWNDATA"/> pre-set — the shape every
        ///     allocating scalar constructor stores (a fresh scalar storage owns its buffer; NumPy's
        ///     <c>ctors.c</c> raises <c>NPY_ARRAY_OWNDATA</c> on every allocating constructor). Pre-built
        ///     once so the hot scalar ctors pay no per-construction flag fixup.
        /// </summary>
        private static readonly Shape ScalarOwnedShape = Shape.Scalar.WithFlags(flagsToSet: ArrayFlags.OWNDATA);

        /// <summary>
        ///     A fresh C-contiguous vector shape carrying <see cref="ArrayFlags.OWNDATA"/> — stored by the
        ///     allocating managed-array constructors, which copy <c>values</c> into brand-new unmanaged
        ///     memory this storage owns.
        /// </summary>
        private static Shape OwnedVectorShape(long length) => new Shape(length).WithFlags(flagsToSet: ArrayFlags.OWNDATA);

        /// <summary>
        ///     Scalar constructor
        /// </summary>
        private unsafe UnmanagedStorage(IArraySlice values)
        {
            _shape = ScalarOwnedShape;
            _dtype = (_typecode = values.TypeCode).AsType();
            Address = (byte*)values.Address;
            Count = 1;
            SetInternalArray(values);
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="dtype"/>.
        /// </summary>
        /// <param name="dtype">The type of this storage</param>
        /// <remarks>Usually <see cref="Allocate(NumSharp.Shape,System.Type)"/> is called after this constructor.</remarks>
        public UnmanagedStorage(Type dtype)
        {
            _dtype = dtype ?? throw new ArgumentNullException(nameof(dtype));
            _typecode = dtype.GetTypeCode();
        }

        /// <summary>
        ///     Creates an empty storage of type <paramref name="typeCode"/>.
        /// </summary>
        /// <param name="typeCode">The type of this storage</param>
        /// <remarks>Usually <see cref="Allocate(NumSharp.Shape,System.Type)"/> is called after this constructor.</remarks>
        public UnmanagedStorage(NPTypeCode typeCode)
        {
            if (typeCode == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(typeCode));

            _dtype = typeCode.AsType();
            _typecode = typeCode;
        }

        private UnmanagedStorage(object value)
        {
            _Allocate(Shape.Scalar, ArraySlice.Scalar(value));
        }

        /// <summary>
        ///     Wraps given <paramref name="arraySlice"/> in <see cref="UnmanagedStorage"/>.
        /// </summary>
        /// <param name="arraySlice">The slice to wrap </param>
        public UnmanagedStorage(IArraySlice arraySlice, Shape shape)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (shape.size != arraySlice.Count)
                throw new IncorrectShapeException($"Given shape size ({shape.size}) does not match the size of the given storage size ({arraySlice.Count})");

            _Allocate(shape, arraySlice);
        }


        // %foreach supported_dtypes,supported_dtypes_lowercase%
        // public UnmanagedStorage(#2 scalar)
        // {            
            // _dtype = typeof(#1);
            // _typecode = InfoOf<#2>.NPTypeCode;
            // _shape = ScalarOwnedShape;
            // InternalArray = _slices.#1 = ArraySlice.Scalar<#2>(scalar);
            // unsafe
            // {
                // Address = (byte*)_slices.#1.Address;
                // Count = _slices.#1.Count;
            // }
        // }

        // %
        public UnmanagedStorage(bool scalar)
        {
            _dtype = typeof(Boolean);
            _typecode = InfoOf<bool>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Boolean = ArraySlice.Scalar<bool>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Boolean.Address;
                Count = _slices.Boolean.Count;
            }
        }

        public UnmanagedStorage(byte scalar)
        {
            _dtype = typeof(Byte);
            _typecode = InfoOf<byte>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Byte = ArraySlice.Scalar<byte>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Byte.Address;
                Count = _slices.Byte.Count;
            }
        }

        public UnmanagedStorage(short scalar)
        {
            _dtype = typeof(Int16);
            _typecode = InfoOf<short>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Int16 = ArraySlice.Scalar<short>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Int16.Address;
                Count = _slices.Int16.Count;
            }
        }

        public UnmanagedStorage(ushort scalar)
        {
            _dtype = typeof(UInt16);
            _typecode = InfoOf<ushort>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.UInt16 = ArraySlice.Scalar<ushort>(scalar);
            unsafe
            {
                Address = (byte*)_slices.UInt16.Address;
                Count = _slices.UInt16.Count;
            }
        }

        public UnmanagedStorage(int scalar)
        {
            _dtype = typeof(Int32);
            _typecode = InfoOf<int>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Int32 = ArraySlice.Scalar<int>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Int32.Address;
                Count = _slices.Int32.Count;
            }
        }

        public UnmanagedStorage(uint scalar)
        {
            _dtype = typeof(UInt32);
            _typecode = InfoOf<uint>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.UInt32 = ArraySlice.Scalar<uint>(scalar);
            unsafe
            {
                Address = (byte*)_slices.UInt32.Address;
                Count = _slices.UInt32.Count;
            }
        }

        public UnmanagedStorage(long scalar)
        {
            _dtype = typeof(Int64);
            _typecode = InfoOf<long>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Int64 = ArraySlice.Scalar<long>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Int64.Address;
                Count = _slices.Int64.Count;
            }
        }

        public UnmanagedStorage(ulong scalar)
        {
            _dtype = typeof(UInt64);
            _typecode = InfoOf<ulong>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.UInt64 = ArraySlice.Scalar<ulong>(scalar);
            unsafe
            {
                Address = (byte*)_slices.UInt64.Address;
                Count = _slices.UInt64.Count;
            }
        }

        public UnmanagedStorage(char scalar)
        {
            _dtype = typeof(Char);
            _typecode = InfoOf<char>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Char = ArraySlice.Scalar<char>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Char.Address;
                Count = _slices.Char.Count;
            }
        }

        public UnmanagedStorage(double scalar)
        {
            _dtype = typeof(Double);
            _typecode = InfoOf<double>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Double = ArraySlice.Scalar<double>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Double.Address;
                Count = _slices.Double.Count;
            }
        }

        public UnmanagedStorage(float scalar)
        {
            _dtype = typeof(Single);
            _typecode = InfoOf<float>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Single = ArraySlice.Scalar<float>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Single.Address;
                Count = _slices.Single.Count;
            }
        }

        public UnmanagedStorage(decimal scalar)
        {
            _dtype = typeof(Decimal);
            _typecode = InfoOf<decimal>.NPTypeCode;
            _shape = ScalarOwnedShape;
            InternalArray = _slices.Decimal = ArraySlice.Scalar<decimal>(scalar);
            unsafe
            {
                Address = (byte*)_slices.Decimal.Address;
                Count = _slices.Decimal.Count;
            }
        }
        // %foreach supported_dtypes,supported_dtypes_lowercase%
        // public UnmanagedStorage(#1[] values)
        // {            
            // if (values == null)
                // throw new ArgumentNullException(nameof(values));
            // _dtype = typeof(#1);
            // _typecode = _dtype.GetTypeCode();
            // _shape = OwnedVectorShape(values.Length);
            // InternalArray = _slices.#1 = new ArraySlice<#2>(UnmanagedMemoryBlock<#2>.FromArray(values));
            // unsafe
            // {
                // Address = (byte*)_slices.#1.Address;
                // Count = values.Length;
            // }
        // }
        // %
        public UnmanagedStorage(Boolean[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Boolean);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Boolean = new ArraySlice<bool>(UnmanagedMemoryBlock<bool>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Boolean.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Byte[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Byte);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Byte = new ArraySlice<byte>(UnmanagedMemoryBlock<byte>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Byte.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Int16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int16);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Int16 = new ArraySlice<short>(UnmanagedMemoryBlock<short>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Int16.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(UInt16[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt16);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.UInt16 = new ArraySlice<ushort>(UnmanagedMemoryBlock<ushort>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.UInt16.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Int32[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int32);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Int32 = new ArraySlice<int>(UnmanagedMemoryBlock<int>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Int32.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(UInt32[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt32);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.UInt32 = new ArraySlice<uint>(UnmanagedMemoryBlock<uint>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.UInt32.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Int64[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Int64);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Int64 = new ArraySlice<long>(UnmanagedMemoryBlock<long>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Int64.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(UInt64[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(UInt64);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.UInt64 = new ArraySlice<ulong>(UnmanagedMemoryBlock<ulong>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.UInt64.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Char[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Char);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Char = new ArraySlice<char>(UnmanagedMemoryBlock<char>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Char.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Double[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Double);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Double = new ArraySlice<double>(UnmanagedMemoryBlock<double>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Double.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Single[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Single);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Single = new ArraySlice<float>(UnmanagedMemoryBlock<float>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Single.Address;
                Count = values.Length;
            }
        }

        public UnmanagedStorage(Decimal[] values)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));
            _dtype = typeof(Decimal);
            _typecode = _dtype.GetTypeCode();
            _shape = OwnedVectorShape(values.Length);
            InternalArray = _slices.Decimal = new ArraySlice<decimal>(UnmanagedMemoryBlock<decimal>.FromArray(values));
            unsafe
            {
                Address = (byte*)_slices.Decimal.Address;
                Count = values.Length;
            }
        }

        #region Switched Accessing

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected unsafe void SetInternalArray(Array array)
        {
            switch (_typecode)
            {
                // //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                // %foreach supported_dtypes,supported_dtypes_lowercase%
                // case NPTypeCode.#1:
                // {
                    // InternalArray = _slices.#1 = ArraySlice.FromArray<#2>((#2[])array);
                    // Address = (byte*) _slices.#1.Address;
                    // Count = _slices.#1.Count;
                    // break;
                // }
                // %
                // default:
                    // throw new NotSupportedException();
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                {
                    InternalArray = _slices.Boolean = ArraySlice.FromArray<bool>((bool[])array);
                    Address = (byte*)_slices.Boolean.Address;
                    Count = _slices.Boolean.Count;
                    break;
                }

                case NPTypeCode.SByte:
                {
                    InternalArray = _slices.SByte = ArraySlice.FromArray<sbyte>((sbyte[])array);
                    Address = (byte*)_slices.SByte.Address;
                    Count = _slices.SByte.Count;
                    break;
                }

                case NPTypeCode.Byte:
                {
                    InternalArray = _slices.Byte = ArraySlice.FromArray<byte>((byte[])array);
                    Address = (byte*)_slices.Byte.Address;
                    Count = _slices.Byte.Count;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    InternalArray = _slices.Int16 = ArraySlice.FromArray<short>((short[])array);
                    Address = (byte*)_slices.Int16.Address;
                    Count = _slices.Int16.Count;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _slices.UInt16 = ArraySlice.FromArray<ushort>((ushort[])array);
                    Address = (byte*)_slices.UInt16.Address;
                    Count = _slices.UInt16.Count;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _slices.Int32 = ArraySlice.FromArray<int>((int[])array);
                    Address = (byte*)_slices.Int32.Address;
                    Count = _slices.Int32.Count;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _slices.UInt32 = ArraySlice.FromArray<uint>((uint[])array);
                    Address = (byte*)_slices.UInt32.Address;
                    Count = _slices.UInt32.Count;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _slices.Int64 = ArraySlice.FromArray<long>((long[])array);
                    Address = (byte*)_slices.Int64.Address;
                    Count = _slices.Int64.Count;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _slices.UInt64 = ArraySlice.FromArray<ulong>((ulong[])array);
                    Address = (byte*)_slices.UInt64.Address;
                    Count = _slices.UInt64.Count;
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _slices.Char = ArraySlice.FromArray<char>((char[])array);
                    Address = (byte*)_slices.Char.Address;
                    Count = _slices.Char.Count;
                    break;
                }

                case NPTypeCode.Half:
                {
                    InternalArray = _slices.Half = ArraySlice.FromArray<Half>((Half[])array);
                    Address = (byte*)_slices.Half.Address;
                    Count = _slices.Half.Count;
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _slices.Double = ArraySlice.FromArray<double>((double[])array);
                    Address = (byte*)_slices.Double.Address;
                    Count = _slices.Double.Count;
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _slices.Single = ArraySlice.FromArray<float>((float[])array);
                    Address = (byte*)_slices.Single.Address;
                    Count = _slices.Single.Count;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _slices.Decimal = ArraySlice.FromArray<decimal>((decimal[])array);
                    Address = (byte*)_slices.Decimal.Address;
                    Count = _slices.Decimal.Count;
                    break;
                }

                case NPTypeCode.Complex:
                {
                    InternalArray = _slices.Complex = ArraySlice.FromArray<System.Numerics.Complex>((System.Numerics.Complex[])array);
                    Address = (byte*)_slices.Complex.Address;
                    Count = _slices.Complex.Count;
                    break;
                }

                default:
                    throw new NotSupportedException();
            }
        }

        /// <summary>
        ///     Replace internal storage array with given array.
        /// </summary>
        /// <param name="array">The array to set as internal storage</param>
        /// <exception cref="InvalidCastException">When type of <paramref name="array"/> does not match <see cref="DType"/> of this storage</exception>
        protected unsafe void SetInternalArray(IArraySlice array)
        {
            switch (_typecode)
            {
                // //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                // %foreach supported_dtypes,supported_dtypes_lowercase%
                // case NPTypeCode.#1:
                // {
                    // InternalArray = _slices.#1 = (ArraySlice<#2>)array;
                    // Address = (byte*) _slices.#1.Address;
                    // Count = _slices.#1.Count;
                    // break;
                // }
                // %
                // default:
                    // throw new NotSupportedException();
                //Since it is a single assignment, we do not use 'as' casting but rather explicit casting that'll also type-check.
                case NPTypeCode.Boolean:
                {
                    InternalArray = _slices.Boolean = (ArraySlice<bool>)array;
                    Address = (byte*)_slices.Boolean.Address;
                    Count = _slices.Boolean.Count;
                    break;
                }

                case NPTypeCode.SByte:
                {
                    InternalArray = _slices.SByte = (ArraySlice<sbyte>)array;
                    Address = (byte*)_slices.SByte.Address;
                    Count = _slices.SByte.Count;
                    break;
                }

                case NPTypeCode.Byte:
                {
                    InternalArray = _slices.Byte = (ArraySlice<byte>)array;
                    Address = _slices.Byte.Address;
                    Count = _slices.Byte.Count;
                    break;
                }

                case NPTypeCode.Int16:
                {
                    InternalArray = _slices.Int16 = (ArraySlice<short>)array;
                    Address = (byte*)_slices.Int16.Address;
                    Count = _slices.Int16.Count;
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    InternalArray = _slices.UInt16 = (ArraySlice<ushort>)array;
                    Address = (byte*)_slices.UInt16.Address;
                    Count = _slices.UInt16.Count;
                    break;
                }

                case NPTypeCode.Int32:
                {
                    InternalArray = _slices.Int32 = (ArraySlice<int>)array;
                    Address = (byte*)_slices.Int32.Address;
                    Count = _slices.Int32.Count;
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    InternalArray = _slices.UInt32 = (ArraySlice<uint>)array;
                    Address = (byte*)_slices.UInt32.Address;
                    Count = _slices.UInt32.Count;
                    break;
                }

                case NPTypeCode.Int64:
                {
                    InternalArray = _slices.Int64 = (ArraySlice<long>)array;
                    Address = (byte*)_slices.Int64.Address;
                    Count = _slices.Int64.Count;
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    InternalArray = _slices.UInt64 = (ArraySlice<ulong>)array;
                    Address = (byte*)_slices.UInt64.Address;
                    Count = _slices.UInt64.Count;
                    break;
                }

                case NPTypeCode.Char:
                {
                    InternalArray = _slices.Char = (ArraySlice<char>)array;
                    Address = (byte*)_slices.Char.Address;
                    Count = _slices.Char.Count;
                    break;
                }

                case NPTypeCode.Half:
                {
                    InternalArray = _slices.Half = (ArraySlice<Half>)array;
                    Address = (byte*)_slices.Half.Address;
                    Count = _slices.Half.Count;
                    break;
                }

                case NPTypeCode.Double:
                {
                    InternalArray = _slices.Double = (ArraySlice<double>)array;
                    Address = (byte*)_slices.Double.Address;
                    Count = _slices.Double.Count;
                    break;
                }

                case NPTypeCode.Single:
                {
                    InternalArray = _slices.Single = (ArraySlice<float>)array;
                    Address = (byte*)_slices.Single.Address;
                    Count = _slices.Single.Count;
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    InternalArray = _slices.Decimal = (ArraySlice<decimal>)array;
                    Address = (byte*)_slices.Decimal.Address;
                    Count = _slices.Decimal.Count;
                    break;
                }

                case NPTypeCode.Complex:
                {
                    InternalArray = _slices.Complex = (ArraySlice<System.Numerics.Complex>)array;
                    Address = (byte*)_slices.Complex.Address;
                    Count = _slices.Complex.Count;
                    break;
                }

                default:
                    throw new NotSupportedException();
            }
        }

        #endregion

        /// <summary>
        ///     Changes the type of <paramref name="sourceArray"/> to <paramref name="to_dtype"/> if necessary.
        /// </summary>
        /// <param name="sourceArray">The array to change his type</param>
        /// <param name="to_dtype">The type to change to.</param>
        /// <remarks>If the return type is equal to source type, this method does not return a copy.</remarks>
        /// <returns>Returns <see cref="sourceArray"/> or new array with changed type to <see cref="to_dtype"/></returns>
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        protected static Array _ChangeTypeOfArray(Array sourceArray, Type to_dtype)
        {
            if (to_dtype == sourceArray.GetType().GetElementType()) return sourceArray;
            return ArrayConvert.To(sourceArray, to_dtype);
        }

        /// <summary>
        ///     Changes the type of <paramref name="sourceArray"/> to <paramref name="to_dtype"/> if necessary.
        /// </summary>
        /// <param name="sourceArray">The array to change his type</param>
        /// <remarks>If the return type is equal to source type, this method does not return a copy.</remarks>
        /// <returns>Returns <see cref="sourceArray"/> or new array with changed type to <see cref="to_dtype"/></returns>
        [SuppressMessage("ReSharper", "AssignNullToNotNullAttribute")]
        protected static ArraySlice<TOut> _ChangeTypeOfArray<TOut>(IArraySlice sourceArray) where TOut : unmanaged
        {
            if (typeof(TOut) == sourceArray.GetType().GetElementType()) return (ArraySlice<TOut>)sourceArray;
            // SIMD copy-with-cast via NDIter. Was: scalar sourceArray.CastTo<TOut>() loop.
            return (ArraySlice<TOut>)CastSliceViaIterator(sourceArray, InfoOf<TOut>.NPTypeCode);
        }

        #region Allocation

        protected void _Allocate(Shape shape, IArraySlice values)
        {
            //if (shape.IsSliced)
            //{
            //    values = values.Clone();
            //    shape = Shape.Clean();
            //}

            // A fresh storage re-reports ALIGNED even when its shape was borrowed verbatim from a
            // setflags(align: false)-cleared source (GetData's identity subshape, sliced views, …):
            // NumPy recomputes alignment for every new array object, and NumSharp data is always
            // genuinely aligned — the cleared bit belongs to the array it was cleared on alone.
            // (setflags itself mutates via SetShapeUnsafe, which this deliberately does not touch.)
            if ((shape._flags & (int)ArrayFlags.ALIGNED) == 0)
                shape = shape.WithFlags(flagsToSet: ArrayFlags.ALIGNED);

            _shape = shape;
            _typecode = values.TypeCode;

            if (_typecode == NPTypeCode.Empty)
                throw new NotSupportedException($"{values.TypeCode} as a dtype is not supported.");

            _dtype = _typecode.AsType();
            SetInternalArray(values);
            Count = shape.size;

            // OWNDATA mirrors this storage's actual ownership (NumPy ctors.c: allocating constructors
            // raise NPY_ARRAY_OWNDATA; a borrowed shape may carry its previous array's bit either way).
            // View-producers that call _Allocate before wiring _baseStorage (CreateBroadcastedUnsafe)
            // re-sync after that assignment.
            OnReshaped();
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, Type dtype = null)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            var resolved = dtype ?? DType;
            AllocationGuard.CheckDimensions(shape.dimensions, resolved.GetTypeCode());

            _Allocate(FreshWriteable(shape), ArraySlice.Allocate(resolved, shape.size, true));
        }

        // This overload allocates BRAND-NEW owned memory, so its result is always writeable — even when
        // the shape was borrowed from a read-only source (e.g. an elementwise result whose shape is the
        // read-only operand's shape, `mmap_r + 1`). Without this the fresh array would inherit the stale
        // read-only flag and diverge from NumPy. (Views never allocate — they Alias, which correctly
        // inherits read-only.)
        private static Shape FreshWriteable(Shape shape)
            => shape.IsWriteable ? shape : shape.WithFlags(flagsToSet: ArrayFlags.WRITEABLE);

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, Type dtype, bool fillZeros)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            var resolved = dtype ?? DType;
            AllocationGuard.CheckDimensions(shape.dimensions, resolved.GetTypeCode());

            _Allocate(FreshWriteable(shape), ArraySlice.Allocate(resolved, shape.size, fillZeros));
        }

        /// <summary>
        ///     Allocates a new <see cref="Array"/> into memory.
        /// </summary>
        /// <param name="dtype">The type of the Array, if null <see cref="DType"/> is used.</param>
        /// <param name="shape">The shape of the array.</param>
        public void Allocate(Shape shape, NPTypeCode dtype, bool fillZeros)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (dtype == NPTypeCode.Empty)
                throw new ArgumentNullException(nameof(dtype));

            AllocationGuard.CheckDimensions(shape.dimensions, dtype);

            _Allocate(FreshWriteable(shape), ArraySlice.Allocate(dtype, shape.size, fillZeros));
        }

        /// <summary>
        ///     Allocate <paramref name="array"/> into memory.
        /// </summary>
        /// <param name="array">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="array"/></remarks>
        public void Allocate(Array array)
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (array.Length == 0)
                throw new ArgumentException("values can't be an empty array", nameof(array));

            var slice = ArraySlice.FromArray(array);
            _Allocate(Shape.ExtractShape(array), slice);
        }

        /// <summary>
        ///     Assign this <see cref="ArraySlice{T}"/> as the internal array storage and assign <see cref="shape"/> to it.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate<T>(ArraySlice<T> values, Shape shape, bool copy = false) where T : unmanaged
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Count != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            _Allocate(shape, copy ? values.Clone() : values);
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <param name="shape">The shape of the array.</param>
        /// <param name="copy">Should perform a copy of <paramref name="values"/></param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate(IArraySlice values, Shape shape, bool copy = false)
        {
            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Count != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            _Allocate(shape, (IArraySlice)(copy ? values.Clone() : values));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        /// <param name="shape">The shape of given array</param>
        public void Allocate(Array values, Shape shape)
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (shape.IsEmpty)
                throw new ArgumentNullException(nameof(shape));

            if (values.Length != shape.Size)
                throw new ArgumentException($"values.Length does not match shape.Size", nameof(values));

            Type elementType = values.GetType();
            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _Allocate(shape, ArraySlice.FromArray(values));
        }

        /// <summary>
        ///     Allocate <paramref name="values"/> into memory.
        /// </summary>
        /// <param name="values">The array to set as internal data storage</param>
        /// <remarks>Does not copy <paramref name="values"/></remarks>
        public void Allocate<T>(T[] values) where T : unmanaged
        {
            if (values == null)
                throw new ArgumentNullException(nameof(values));

            if (values.Length == 0)
                throw new ArgumentException("values can't be an empty array", nameof(values));

            Shape shape;
            if (values.Rank > 1)
            {
                int[] dim = new int[values.Rank];
                for (int idx = 0; idx < dim.Length; idx++)
                    dim[idx] = values.GetLength(idx);
                shape = new Shape(dim);
            }
            else
            {
                shape = new Shape(values.Length);
            }

            Type elementType = values.GetType();
            // ReSharper disable once PossibleNullReferenceException
            while (elementType.IsArray)
                elementType = elementType.GetElementType();

            _Allocate(shape, ArraySlice.FromArray(values));
        }

        #endregion


        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        public void CopyTo(IntPtr ptr)
        {
            unsafe
            {
                CopyTo(ptr.ToPointer());
            }
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo(void* address)
        {
            // #region Compute

		    // switch (TypeCode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1:
			    // {
				    // CopyTo<#2>((#2*)address);
                    // break;
			    // }

			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }

            // #endregion

            #region Compute

            switch (TypeCode)
            {
                case NPTypeCode.Boolean:
                {
                    CopyTo<bool>((bool*)address);
                    break;
                }

                case NPTypeCode.Byte:
                {
                    CopyTo<byte>((byte*)address);
                    break;
                }

                case NPTypeCode.SByte:
                {
                    CopyTo<sbyte>((sbyte*)address);
                    break;
                }

                case NPTypeCode.Int16:
                {
                    CopyTo<short>((short*)address);
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    CopyTo<ushort>((ushort*)address);
                    break;
                }

                case NPTypeCode.Int32:
                {
                    CopyTo<int>((int*)address);
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    CopyTo<uint>((uint*)address);
                    break;
                }

                case NPTypeCode.Int64:
                {
                    CopyTo<long>((long*)address);
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    CopyTo<ulong>((ulong*)address);
                    break;
                }

                case NPTypeCode.Char:
                {
                    CopyTo<char>((char*)address);
                    break;
                }

                case NPTypeCode.Half:
                {
                    CopyTo<Half>((Half*)address);
                    break;
                }

                case NPTypeCode.Double:
                {
                    CopyTo<double>((double*)address);
                    break;
                }

                case NPTypeCode.Single:
                {
                    CopyTo<float>((float*)address);
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    CopyTo<decimal>((decimal*)address);
                    break;
                }

                case NPTypeCode.Complex:
                {
                    CopyTo<System.Numerics.Complex>((System.Numerics.Complex*)address);
                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe void CopyTo(IMemoryBlock block)
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given memory block because this storage count is larger than the given memory block's length.");

            // #region Compute

		    // switch (TypeCode)
		    // {
			    // %foreach supported_dtypes,supported_dtypes_lowercase%
			    // case NPTypeCode.#1:
			    // {
				    // CopyTo<#2>((#2*)slice.Address);
                    // break;
			    // }

			    // %
			    // default:
				    // throw new NotSupportedException();
		    // }

            // #endregion

            #region Compute

            switch (TypeCode)
            {
                case NPTypeCode.Boolean:
                {
                    CopyTo<bool>((bool*)block.Address);
                    break;
                }

                case NPTypeCode.Byte:
                {
                    CopyTo<byte>((byte*)block.Address);
                    break;
                }

                case NPTypeCode.SByte:
                {
                    CopyTo<sbyte>((sbyte*)block.Address);
                    break;
                }

                case NPTypeCode.Int16:
                {
                    CopyTo<short>((short*)block.Address);
                    break;
                }

                case NPTypeCode.UInt16:
                {
                    CopyTo<ushort>((ushort*)block.Address);
                    break;
                }

                case NPTypeCode.Int32:
                {
                    CopyTo<int>((int*)block.Address);
                    break;
                }

                case NPTypeCode.UInt32:
                {
                    CopyTo<uint>((uint*)block.Address);
                    break;
                }

                case NPTypeCode.Int64:
                {
                    CopyTo<long>((long*)block.Address);
                    break;
                }

                case NPTypeCode.UInt64:
                {
                    CopyTo<ulong>((ulong*)block.Address);
                    break;
                }

                case NPTypeCode.Char:
                {
                    CopyTo<char>((char*)block.Address);
                    break;
                }

                case NPTypeCode.Half:
                {
                    CopyTo<Half>((Half*)block.Address);
                    break;
                }

                case NPTypeCode.Double:
                {
                    CopyTo<double>((double*)block.Address);
                    break;
                }

                case NPTypeCode.Single:
                {
                    CopyTo<float>((float*)block.Address);
                    break;
                }

                case NPTypeCode.Decimal:
                {
                    CopyTo<decimal>((decimal*)block.Address);
                    break;
                }

                case NPTypeCode.Complex:
                {
                    CopyTo<System.Numerics.Complex>((System.Numerics.Complex*)block.Address);
                    break;
                }

                default:
                    throw new NotSupportedException();
            }

            #endregion

        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address (using <see cref="Count"/>).
        /// </summary>
        /// <param name="block">The block to copy to.</param>
        public unsafe void CopyTo<T>(IMemoryBlock<T> block) where T : unmanaged
        {
            if (block.TypeCode != _typecode)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > block.Count)
                throw new ArgumentOutOfRangeException(nameof(block), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            CopyTo<T>(block.Address);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given address.
        /// </summary>
        /// <param name="address">The address to copy to.</param>
        public unsafe void CopyTo<T>(T* address) where T : unmanaged
        {
            if (address == (T*)0)
                throw new ArgumentNullException(nameof(address));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (!Shape.IsContiguous)
            {
                var dst = ArraySlice.Wrap<T>(address, Count);
                NDIter.Copy(new UnmanagedStorage(dst, Shape.Clean()), this);
                return;
            }

            // Fast path for contiguous - account for offset (sliced views)
            var bytesCount = Count * InfoOf<T>.Size;
            var srcAddress = Address + Shape.offset * InfoOf<T>.Size;
            Buffer.MemoryCopy(srcAddress, address, bytesCount, bytesCount);
        }

        /// <summary>
        ///     Copies the entire contents of this storage to given array.
        /// </summary>
        /// <param name="array">The array to copy to.</param>
        public unsafe void CopyTo<T>(T[] array) where T : unmanaged
        {
            if (array == null)
                throw new ArgumentNullException(nameof(array));

            if (typeof(T) != _dtype)
                throw new InvalidCastException("Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.");

            if (Count > array.Length)
                throw new ArgumentOutOfRangeException(nameof(array), $"Unable to copy from this storage to given array because this storage count is larger than the given array length.");

            fixed (T* dst = array)
            {
                CopyTo<T>(dst);
            }
        }

        [MethodImpl(Optimize)]
        public unsafe T[] ToArray<T>() where T : unmanaged
        {
            if (typeof(T).GetTypeCode() != InternalArray.TypeCode)
                throw new ArrayTypeMismatchException($"The given type argument '{typeof(T).Name}' doesn't match the type of the internal data '{InternalArray.TypeCode}'");

            var src = (T*)Address;

            // .NET arrays are limited to int32 indexing
            if (Shape.Size > int.MaxValue)
                throw new InvalidOperationException($"Array size {Shape.Size} exceeds int.MaxValue. Use ToArraySlice() for large arrays.");

            var ret = new T[(int)Shape.Size];

            // NumPy-aligned: For contiguous shapes, use fast memory copy.
            // Must account for shape.offset which indicates the starting position in the buffer.
            if (Shape.IsContiguous)
            {
                // Adjust source pointer by offset for sliced views
                var srcWithOffset = src + Shape.offset;
                fixed (T* dst = ret)
                {
                    var len = sizeof(T) * ret.Length;
                    Buffer.MemoryCopy(srcWithOffset, dst, len, len);
                }
            }
            else
            {
                var incr = new ValueCoordinatesIncrementor(Shape.dimensions);
                long[] current = incr.Index;
                int i = 0;
                ref Shape shape = ref ShapeReference;
                do ret[i++] = src[shape.GetOffset(current)];
                while (incr.Next() != null);
            }

            return ret;
        }
    }
}
