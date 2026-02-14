using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Aliasing

        /// <summary>
        /// Creates an alias (view) of this storage that shares the same underlying memory.
        /// </summary>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage.
        /// The returned storage's <see cref="_baseStorage"/> points to the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner:
        /// <list type="bullet">
        ///   <item>If this storage owns its data: <c>alias._baseStorage = this</c></item>
        ///   <item>If this storage is a view: <c>alias._baseStorage = this._baseStorage</c></item>
        /// </list>
        /// This ensures all views in a chain point to the original owner, not intermediate views.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public UnmanagedStorage Alias()
        {
            var r = new UnmanagedStorage();
            r._shape = _shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = _shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            return r;
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different shape.
        /// </summary>
        /// <param name="shape">The shape for the alias. Should be compatible with the storage size (not validated).</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage but has
        /// the specified shape. The returned storage's <see cref="_baseStorage"/> points to
        /// the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Shape Compatibility:</b> This method does NOT validate that the shape is
        /// compatible with the storage size. Use with caution.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public UnmanagedStorage Alias(Shape shape)
        {
            var r = new UnmanagedStorage();
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);

            r._shape = shape;
            r.Count = shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            return r;
        }

        /// <summary>
        /// Creates an alias (view) of this storage with a different shape (by reference).
        /// </summary>
        /// <param name="shape">The shape for the alias. Should be compatible with the storage size (not validated).</param>
        /// <returns>
        /// A new <see cref="UnmanagedStorage"/> that shares memory with this storage but has
        /// the specified shape. The returned storage's <see cref="_baseStorage"/> points to
        /// the ultimate owner.
        /// </returns>
        /// <remarks>
        /// <para>
        /// <b>Memory Sharing:</b> The alias shares the same <see cref="InternalArray"/> and
        /// underlying memory. Modifications through the alias affect the original data.
        /// </para>
        /// <para>
        /// <b>Shape Compatibility:</b> This method does NOT validate that the shape is
        /// compatible with the storage size. Use with caution.
        /// </para>
        /// <para>
        /// <b>Base Tracking:</b> Sets <c>_baseStorage</c> to chain to the ultimate owner.
        /// </para>
        /// </remarks>
        /// <seealso cref="Clone"/>
        public UnmanagedStorage Alias(ref Shape shape)
        {
            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = shape.size; //incase shape is sliced
            r._baseStorage = _baseStorage ?? this;
            return r;
        }

        #endregion

        #region Casting

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast<T>() where T : unmanaged
        {
            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeof(T));

            if (_dtype == typeof(T))
                return Clone();

            //this also handles slices
            return new UnmanagedStorage((ArraySlice<T>)InternalArray.CastTo<T>(), _shape.Clone(true, true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(NPTypeCode typeCode)
        {
            if (_shape.IsEmpty)
                return new UnmanagedStorage(typeCode);

            if (_typecode == typeCode)
                return Clone();

            //this also handles slices
            return new UnmanagedStorage((IArraySlice)InternalArray.CastTo(typeCode), _shape.Clone(true, true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Always copies, If dtype==typeof(T) then a <see cref="Clone"/> is returned.</remarks>
        public UnmanagedStorage Cast(Type dtype)
        {
            return Cast(dtype.GetTypeCode());
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype only if necessary.
        /// </summary>
        /// <typeparam name="T">The dtype to convert to</typeparam>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <typeparamref name="T"/></remarks>
        public UnmanagedStorage CastIfNecessary<T>() where T : unmanaged
        {
            if (_shape.IsEmpty || _dtype == typeof(T))
                return this;

            //this also handles slices
            return new UnmanagedStorage((ArraySlice<T>)InternalArray.CastTo<T>(), _shape.Clone(true, true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype only if necessary
        /// </summary>
        /// <param name="typeCode">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(NPTypeCode typeCode)
        {
            if (_shape.IsEmpty || _typecode == typeCode)
                return this;

            //this also handles slices
            return new UnmanagedStorage((IArraySlice)InternalArray.CastTo(typeCode), _shape.Clone(true, true, true));
        }

        /// <summary>
        ///     Return a casted <see cref="UnmanagedStorage"/> to a specific dtype.
        /// </summary>
        /// <param name="dtype">The dtype to convert to</param>
        /// <returns>A copy of this <see cref="UnmanagedStorage"/> casted to a specific dtype.</returns>
        /// <remarks>Copies only if dtypes does not match <paramref name="typeCode"/></remarks>
        public UnmanagedStorage CastIfNecessary(Type dtype)
        {
            return CastIfNecessary(dtype.GetTypeCode());
        }

        #endregion

        #region Cloning

        /// <summary>
        ///     Clone internal storage and get reference to it
        /// </summary>
        /// <returns>reference to cloned storage as System.Array</returns>
        public IArraySlice CloneData()
        {
            // Contiguous shapes can copy directly from memory.
            // Must account for offset - slice the internal array at the correct position.
            if (_shape.IsContiguous)
            {
                if (_shape.offset == 0)
                    return InternalArray.Clone();
                else
                    return InternalArray.Slice(_shape.offset, _shape.size).Clone();
            }

            if (_shape.IsScalar)
                return ArraySlice.Scalar(GetValue(0), _typecode);

            //Linear copy of all the sliced items (non-contiguous: broadcast, stepped, transposed).

            var ret = ArraySlice.Allocate(InternalArray.TypeCode, _shape.size, false);
            MultiIterator.Assign(new UnmanagedStorage(ret, _shape.Clean()), this);

            return ret;
        }

        /// <summary>
        ///     Get all elements from cloned storage as <see cref="ArraySlice{T}"/> and cast if necessary.
        /// </summary>
        /// <typeparam name="T">cloned storgae dtype</typeparam>
        /// <returns>reference to cloned storage and casted (if necessary) as <see cref="ArraySlice{T}"/></returns>
        public ArraySlice<T> CloneData<T>() where T : unmanaged
        {
            var cloned = CloneData();
            if (cloned.TypeCode != InfoOf<T>.NPTypeCode)
                return (ArraySlice<T>)cloned.CastTo<T>();

            return (ArraySlice<T>)cloned;
        }

        /// <summary>
        ///     Perform a complete copy of this <see cref="UnmanagedStorage"/> and <see cref="InternalArray"/>.
        /// </summary>
        /// <remarks>If shape is sliced, discards any slicing properties but copies only the sliced data</remarks>
        public UnmanagedStorage Clone() => new UnmanagedStorage(CloneData(), _shape.Clone(true, true, true));

        object ICloneable.Clone() => Clone();

        #endregion
    }
}
