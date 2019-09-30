using System;
using NumSharp.Backends.Unmanaged;
using NumSharp.Utilities;

namespace NumSharp.Backends
{
    public partial class UnmanagedStorage
    {
        #region Aliasing

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage.
        /// </summary>
        public UnmanagedStorage Alias()
        {
            var r = new UnmanagedStorage();
            r._shape = _shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = _shape.size; //incase shape is sliced
            return r;
        }

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage with a specific shape.
        /// </summary>
        /// <remarks>Doesn't check if Shape matches the internal storage.</remarks>
        public UnmanagedStorage Alias(Shape shape)
        {
            var r = new UnmanagedStorage();
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);

            r._shape = shape;
            r.Count = shape.size; //incase shape is sliced
            return r;
        }

        /// <summary>
        ///     Creates an alias to this UnmanagedStorage with a specific shape.
        /// </summary>
        /// <remarks>Doesn't check if Shape matches the internal storage.</remarks>
        public UnmanagedStorage Alias(ref Shape shape)
        {
            var r = new UnmanagedStorage();
            r._shape = shape;
            r._typecode = _typecode;
            r._dtype = _dtype;
            if (InternalArray != null)
                r.SetInternalArray(InternalArray);
            r.Count = shape.size; //incase shape is sliced
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
            //Incase shape is not sliced, we can copy the internal buffer.
            if (!_shape.IsSliced && !_shape.IsBroadcasted)
                return InternalArray.Clone();

            if (_shape.IsScalar)
                return ArraySlice.Scalar(GetValue(0), _typecode);

            //Linear copy of all the sliced items.

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
